"""
Tests for the compiled objective-function cache in the dew point solver.

``meteocalc.dewpoint._solver_method`` keeps one ``@njit`` objective per
``(vapor equation class, surface type)`` in ``_OBJECTIVE_CACHE``. Building a
fresh closure on every call gives Numba a new function type, which forces
rapid-roots to compile a new solver specialisation each time. These tests check
that:

1. The cache key separates everything that changes the compiled constants
   (equation class, surface type) and nothing else (equation instance, how the
   surface type was spelled).
2. A cached objective evaluates exactly what an uncached one would, with the
   constants for its own key.
3. Solver results are identical whether the cache is cold, warm, bypassed, or
   shared between interleaved equations and surface types.
4. A warm call compiles nothing, and rapid-roots does not accumulate solver
   specialisations across repeated calls.

Tests that need an empty cache swap in a fresh dict with ``monkeypatch`` so the
module-level cache is restored afterwards and other tests are not affected.
"""

import numpy as np
import pytest
from numba.core import event as numba_event
from numpy.testing import assert_allclose, assert_array_equal
from rapid_roots.solvers import RootSolvers

import meteocalc.dewpoint._solver_method as solver_module
from meteocalc.dewpoint import Dewpoint
from meteocalc.dewpoint._solver_method import (
    get_dewpoint_objective_function,
    get_dewpoint_using_solver,
)
from meteocalc.shared._shared_enums import SurfaceType
from meteocalc.vapor import Vapor
from meteocalc.vapor._jit_equations import (
    _goff_gratch_scalar,
    _hyland_wexler_scalar,
)
from meteocalc.vapor._vapor_constants import (
    GOFF_GRATCH_ICE,
    GOFF_GRATCH_WATER,
    HYLAND_WEXLER_ICE,
    HYLAND_WEXLER_WATER,
)
from meteocalc.vapor._vapor_equations import (
    GoffGratchEquation,
    HylandWexlerEquation,
)

# ==============================================================================
# Inputs and helpers
# ==============================================================================

# Combinations the dew point solver supports. Bolton is excluded because the
# solver cannot build an objective for it yet (see TestKnownLimitations).
SUPPORTED = [
    ("goff_gratch", SurfaceType.WATER),
    ("goff_gratch", SurfaceType.ICE),
    ("hyland_wexler", SurfaceType.WATER),
    ("hyland_wexler", SurfaceType.ICE),
]

# Temperatures inside each surface's valid range, so no bound warnings fire.
WATER_TEMPS = np.linspace(275.15, 310.15, 64)
ICE_TEMPS = np.linspace(235.15, 270.15, 64)

# RH strictly below 1: at RH = 1 the root sits on the bracket edge, which is a
# separate known issue and not what these tests are about.
RH_VALUES = np.linspace(0.05, 0.98, 64)


# Independent reference: the JIT function and constants each cache key must
# use, taken straight from the vapor module rather than from the equation
# object the objective was built from.
REFERENCE = {
    ("goff_gratch", SurfaceType.WATER): (_goff_gratch_scalar, GOFF_GRATCH_WATER),
    ("goff_gratch", SurfaceType.ICE): (_goff_gratch_scalar, GOFF_GRATCH_ICE),
    ("hyland_wexler", SurfaceType.WATER): (_hyland_wexler_scalar, HYLAND_WEXLER_WATER),
    ("hyland_wexler", SurfaceType.ICE): (_hyland_wexler_scalar, HYLAND_WEXLER_ICE),
}


def _reference_e_sat(name, surface_type, temp_k):
    scalar_func, constants = REFERENCE[(name, SurfaceType(surface_type))]
    return scalar_func(float(temp_k), *constants)


def _temps_for(surface_type):
    if SurfaceType(surface_type) == SurfaceType.WATER:
        return WATER_TEMPS
    return ICE_TEMPS


def _equation(name, surface_type):
    return Vapor.get_equation(equation=name, phase=surface_type)


def _solve(name, surface_type, temps=None, rh=None):
    temps = _temps_for(surface_type) if temps is None else temps
    rh = RH_VALUES if rh is None else rh
    return get_dewpoint_using_solver(
        temp_k=temps, rh=rh, surface_type=surface_type, vapor_equation=name
    )


def _count_compilations(func, *args, **kwargs):
    """Run ``func`` and return the number of Numba compile events it caused."""
    with numba_event.install_recorder("numba:compile") as recorder:
        func(*args, **kwargs)
    return len(recorder.buffer)


class _NoStoreDict(dict):
    """A cache that never keeps anything, to simulate the uncached code path."""

    def __setitem__(self, key, value):
        pass


@pytest.fixture
def empty_cache(monkeypatch):
    """Give the test an empty objective cache and restore the real one after."""
    cache = {}
    monkeypatch.setattr(solver_module, "_OBJECTIVE_CACHE", cache)
    return cache


@pytest.fixture
def no_store_cache(monkeypatch):
    """Make every call build a new objective, as before caching was added."""
    monkeypatch.setattr(solver_module, "_OBJECTIVE_CACHE", _NoStoreDict())


# ==============================================================================
# 1. Cache keying
# ==============================================================================


class TestCacheKeying:
    """What goes into the cache, under which key."""

    def test_first_call_stores_one_entry_under_class_and_surface(self, empty_cache):
        equation = GoffGratchEquation(surface_type="water")
        objective = get_dewpoint_objective_function(equation)

        assert list(empty_cache) == [(GoffGratchEquation, SurfaceType.WATER)]
        assert empty_cache[(GoffGratchEquation, SurfaceType.WATER)] is objective

    def test_objective_is_a_numba_dispatcher(self, empty_cache):
        objective = get_dewpoint_objective_function(
            GoffGratchEquation(surface_type="water")
        )
        # Only a Numba dispatcher can be passed into the compiled solver; a
        # plain Python function would mean the cache stores the wrong thing.
        assert hasattr(objective, "overloads")
        assert hasattr(objective, "py_func")

    def test_same_instance_returns_same_object(self, empty_cache):
        equation = GoffGratchEquation(surface_type="water")
        assert get_dewpoint_objective_function(
            equation
        ) is get_dewpoint_objective_function(equation)

    def test_new_instances_of_same_class_and_surface_share_objective(self, empty_cache):
        # The public API builds a new equation instance on every call, so the
        # cache only helps if it is keyed by class rather than by instance.
        first = get_dewpoint_objective_function(_equation("goff_gratch", "water"))
        second = get_dewpoint_objective_function(_equation("goff_gratch", "water"))
        assert first is second
        assert len(empty_cache) == 1

    @pytest.mark.parametrize(
        "phase", ["water", "WATER", "Water", SurfaceType.WATER], ids=repr
    )
    def test_surface_type_spelling_does_not_split_the_cache(self, empty_cache, phase):
        reference = get_dewpoint_objective_function(
            GoffGratchEquation(surface_type=SurfaceType.WATER)
        )
        objective = get_dewpoint_objective_function(
            Vapor.get_equation(equation="goff_gratch", phase=phase)
        )
        assert objective is reference
        assert len(empty_cache) == 1

    def test_water_and_ice_get_separate_objectives(self, empty_cache):
        water = get_dewpoint_objective_function(_equation("goff_gratch", "water"))
        ice = get_dewpoint_objective_function(_equation("goff_gratch", "ice"))

        assert water is not ice
        assert set(empty_cache) == {
            (GoffGratchEquation, SurfaceType.WATER),
            (GoffGratchEquation, SurfaceType.ICE),
        }

    def test_equation_classes_get_separate_objectives(self, empty_cache):
        goff_gratch = get_dewpoint_objective_function(_equation("goff_gratch", "water"))
        hyland_wexler = get_dewpoint_objective_function(
            _equation("hyland_wexler", "water")
        )
        assert goff_gratch is not hyland_wexler
        assert set(empty_cache) == {
            (GoffGratchEquation, SurfaceType.WATER),
            (HylandWexlerEquation, SurfaceType.WATER),
        }

    def test_every_supported_combination_gets_its_own_entry(self, empty_cache):
        objectives = [
            get_dewpoint_objective_function(_equation(name, surface))
            for name, surface in SUPPORTED
        ]
        assert len({id(objective) for objective in objectives}) == len(SUPPORTED)
        assert len(empty_cache) == len(SUPPORTED)

    def test_repeated_calls_do_not_grow_the_cache(self, empty_cache):
        for _ in range(25):
            for name, surface in SUPPORTED:
                get_dewpoint_objective_function(_equation(name, surface))
        assert len(empty_cache) == len(SUPPORTED)

    def test_cache_hit_does_not_rebuild(self, empty_cache, monkeypatch):
        calls = []
        original = GoffGratchEquation.get_jit_scalar_func

        def counting_get_jit_scalar_func(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(
            GoffGratchEquation, "get_jit_scalar_func", counting_get_jit_scalar_func
        )

        for _ in range(10):
            get_dewpoint_objective_function(_equation("goff_gratch", "water"))

        assert len(calls) == 1

    def test_failed_build_is_not_cached(self, empty_cache, monkeypatch):
        original = GoffGratchEquation.get_jit_scalar_func
        state = {"fail": True}

        def flaky_get_jit_scalar_func(self):
            if state["fail"]:
                raise RuntimeError("simulated build failure")
            return original(self)

        monkeypatch.setattr(
            GoffGratchEquation, "get_jit_scalar_func", flaky_get_jit_scalar_func
        )

        with pytest.raises(RuntimeError, match="simulated build failure"):
            get_dewpoint_objective_function(_equation("goff_gratch", "water"))
        assert empty_cache == {}

        state["fail"] = False
        objective = get_dewpoint_objective_function(_equation("goff_gratch", "water"))
        assert empty_cache[(GoffGratchEquation, SurfaceType.WATER)] is objective


# ==============================================================================
# 2. Values computed by cached objectives
# ==============================================================================


class TestCachedObjectiveValues:
    """A cached objective must compute e_s(x) - e_actual for its own key."""

    @pytest.mark.parametrize("name, surface", SUPPORTED)
    def test_objective_matches_reference_formula(self, name, surface):
        objective = get_dewpoint_objective_function(_equation(name, surface))

        e_actual = 3.21
        for x in _temps_for(surface)[::8]:
            expected = _reference_e_sat(name, surface, x) - e_actual
            assert_allclose(objective(float(x), e_actual), expected, rtol=1e-13)

    @pytest.mark.parametrize("name, surface", SUPPORTED)
    def test_objective_is_zero_at_saturation(self, name, surface):
        objective = get_dewpoint_objective_function(_equation(name, surface))

        for x in _temps_for(surface)[::8]:
            e_sat = _reference_e_sat(name, surface, x)
            assert abs(objective(float(x), e_sat)) <= 1e-12 * e_sat

    @pytest.mark.parametrize("name, surface", SUPPORTED)
    def test_objective_built_from_a_later_instance_is_unchanged(
        self, empty_cache, name, surface
    ):
        # The first instance builds the objective; later instances are served
        # the cached one. Its values must still match the reference for the key.
        get_dewpoint_objective_function(_equation(name, surface))
        later = get_dewpoint_objective_function(_equation(name, surface))

        for x in _temps_for(surface)[::16]:
            assert_allclose(
                later(float(x), 0.0), _reference_e_sat(name, surface, x), rtol=1e-13
            )

    @pytest.mark.parametrize("name", ["goff_gratch", "hyland_wexler"])
    def test_water_and_ice_objectives_use_their_own_constants(self, name):
        water_objective = get_dewpoint_objective_function(_equation(name, "water"))
        ice_objective = get_dewpoint_objective_function(_equation(name, "ice"))

        # At -10 °C saturation over water is about 10% higher than over ice. If
        # the two keys ever collided, both objectives would return one value.
        temp_k = 263.15
        e_water = water_objective(temp_k, 0.0)
        e_ice = ice_objective(temp_k, 0.0)
        assert_allclose(e_water / e_ice, 1.1, rtol=0.01)

        assert_allclose(e_water, _reference_e_sat(name, "water", temp_k), rtol=1e-13)
        assert_allclose(e_ice, _reference_e_sat(name, "ice", temp_k), rtol=1e-13)

    def test_equation_classes_compute_different_values(self):
        goff_gratch = get_dewpoint_objective_function(_equation("goff_gratch", "water"))
        hyland_wexler = get_dewpoint_objective_function(
            _equation("hyland_wexler", "water")
        )
        # The two formulations agree closely but not exactly (about 0.13% at
        # 20 °C); identical output would mean one class got the other's objective.
        e_gg = goff_gratch(293.15, 0.0)
        e_hw = hyland_wexler(293.15, 0.0)
        assert e_gg != e_hw
        assert_allclose(e_gg, e_hw, rtol=5e-3)
        assert_allclose(e_gg, _reference_e_sat("goff_gratch", "water", 293.15))
        assert_allclose(e_hw, _reference_e_sat("hyland_wexler", "water", 293.15))


# ==============================================================================
# 3. Solver results with the cache
# ==============================================================================


class TestSolverResultsWithCache:
    """The cache must not change any result, bit for bit."""

    def test_cold_and_warm_cache_give_identical_results(self, empty_cache):
        cold_roots, cold_iters, cold_converged = _solve("goff_gratch", "water")
        warm_roots, warm_iters, warm_converged = _solve("goff_gratch", "water")

        assert cold_converged.all()
        assert_array_equal(warm_roots, cold_roots)
        assert_array_equal(warm_iters, cold_iters)
        assert_array_equal(warm_converged, cold_converged)

    @pytest.mark.parametrize("name, surface", SUPPORTED)
    def test_results_are_physically_valid(self, name, surface):
        temps = _temps_for(surface)
        roots, _, converged = _solve(name, surface)

        assert converged.all()
        assert np.all(roots < temps)

        # The root must satisfy e_s(Td) = rh * e_s(T) using the formula for its
        # own key. Brent stops at a 1e-6 K bracket, about 1e-7 relative in e_s.
        for td, temp_k, rh in zip(roots, temps, RH_VALUES):
            e_dew = _reference_e_sat(name, surface, td)
            e_actual = rh * _reference_e_sat(name, surface, temp_k)
            assert_allclose(e_dew, e_actual, rtol=1e-6)

    def test_interleaved_surface_types_match_isolated_runs(self):
        water_first = _solve("goff_gratch", "water")[0]
        ice_first = _solve("goff_gratch", "ice")[0]
        water_again = _solve("goff_gratch", "water")[0]
        ice_again = _solve("goff_gratch", "ice")[0]

        assert_array_equal(water_again, water_first)
        assert_array_equal(ice_again, ice_first)

    @pytest.mark.filterwarnings("ignore:Temperature is outside the valid range")
    def test_frost_point_differs_from_dew_point_on_the_same_inputs(self):
        # Same air below freezing, two surfaces. Separate cache entries must
        # give the frost point above the dew point.
        temps = np.full(16, 263.15)
        rh = np.linspace(0.3, 0.9, 16)
        e_actual = rh * Vapor.get_vapor_saturation(
            temp_k=temps, phase="water", equation="goff_gratch"
        )

        roots = {}
        for surface in ("water", "ice"):
            objective = get_dewpoint_objective_function(
                _equation("goff_gratch", surface)
            )
            roots[surface], _, converged = RootSolvers.get_root(
                func=objective,
                a=temps - 100,
                b=temps,
                func_params=e_actual.reshape(-1, 1).copy(),
                main_solver="brent",
                use_backup=True,
                backup_solvers=["bisection"],
            )
            assert converged.all()

        assert np.all(roots["ice"] > roots["water"])

    def test_interleaved_equations_match_isolated_runs(self):
        goff_gratch_first = _solve("goff_gratch", "water")[0]
        hyland_wexler_first = _solve("hyland_wexler", "water")[0]
        goff_gratch_again = _solve("goff_gratch", "water")[0]
        hyland_wexler_again = _solve("hyland_wexler", "water")[0]

        assert_array_equal(goff_gratch_again, goff_gratch_first)
        assert_array_equal(hyland_wexler_again, hyland_wexler_first)
        assert not np.array_equal(goff_gratch_first, hyland_wexler_first)
        assert_allclose(goff_gratch_first, hyland_wexler_first, atol=0.05)

    def test_scalar_and_array_calls_agree(self):
        roots, _, _ = _solve("goff_gratch", "water")
        for i in (0, 17, 63):
            scalar_root, _, scalar_converged = get_dewpoint_using_solver(
                temp_k=float(WATER_TEMPS[i]),
                rh=float(RH_VALUES[i]),
                surface_type="water",
                vapor_equation="goff_gratch",
            )
            assert isinstance(scalar_root, float)
            assert scalar_converged
            assert scalar_root == roots[i]

    def test_public_api_uses_the_same_cache_entries(self, empty_cache):
        dewpoint = Dewpoint.get_dewpoint_solver(WATER_TEMPS, RH_VALUES)
        frostpoint = Dewpoint.get_frostpoint_solver(ICE_TEMPS, RH_VALUES)

        assert set(empty_cache) == {
            (GoffGratchEquation, SurfaceType.WATER),
            (GoffGratchEquation, SurfaceType.ICE),
        }
        assert_array_equal(dewpoint, _solve("goff_gratch", "water")[0])
        assert_array_equal(frostpoint, _solve("goff_gratch", "ice")[0])
        assert len(empty_cache) == 2

    @pytest.mark.slow
    def test_cached_results_match_uncached_results(self, monkeypatch):
        cached = _solve("hyland_wexler", "ice")

        monkeypatch.setattr(solver_module, "_OBJECTIVE_CACHE", _NoStoreDict())
        uncached = _solve("hyland_wexler", "ice")

        for cached_part, uncached_part in zip(cached, uncached):
            assert_array_equal(cached_part, uncached_part)


# ==============================================================================
# 4. Compilation
# ==============================================================================


class TestNoRecompilation:
    """The point of the cache: warm calls must not compile anything."""

    @pytest.mark.parametrize("name, surface", SUPPORTED)
    def test_warm_array_call_compiles_nothing(self, name, surface):
        _solve(name, surface)  # warm up
        assert _count_compilations(_solve, name, surface) == 0

    def test_warm_scalar_call_compiles_nothing(self):
        get_dewpoint_using_solver(290.0, 0.5, "water", "goff_gratch")  # warm up
        compilations = _count_compilations(
            get_dewpoint_using_solver, 285.0, 0.7, "water", "goff_gratch"
        )
        assert compilations == 0

    def test_warm_public_api_calls_compile_nothing(self):
        Dewpoint.get_dewpoint_solver(WATER_TEMPS, RH_VALUES)  # warm up
        Dewpoint.get_frostpoint_solver(ICE_TEMPS, RH_VALUES)

        def calls():
            Dewpoint.get_dewpoint_solver(WATER_TEMPS, RH_VALUES)
            Dewpoint.get_frostpoint_solver(ICE_TEMPS, RH_VALUES)
            Dewpoint.get_dewpoint_solver(293.15, 0.6)

        assert _count_compilations(calls) == 0

    def test_solver_specialisations_do_not_grow(self):
        try:
            from rapid_roots.solvers._codegen import generate_vectorised_solver
            from rapid_roots.solvers._jit_solvers import _brent_scalar
        except ImportError:
            pytest.skip("rapid-roots internals have moved")

        for name, surface in SUPPORTED:  # warm up every combination
            _solve(name, surface)

        brent_solver = generate_vectorised_solver(_brent_scalar, 1, "bracket")
        specialisations = len(brent_solver.overloads)
        assert specialisations >= 1  # confirms this is the solver in use

        for _ in range(10):
            for name, surface in SUPPORTED:
                _solve(name, surface)

        assert len(brent_solver.overloads) == specialisations

    @pytest.mark.slow
    def test_without_the_cache_every_call_compiles(self, no_store_cache):
        # Negative control: proves the compile counter can see the regression
        # the cache prevents, so a zero above is meaningful.
        _solve("goff_gratch", "water")
        assert _count_compilations(_solve, "goff_gratch", "water") > 0


# ==============================================================================
# 5. Known limitations
# ==============================================================================


class TestKnownLimitations:
    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            "BoltonEquation.get_constants() returns None, and the objective "
            "builder calls tuple() on it. Remove this marker once fixed."
        ),
    )
    def test_bolton_objective_can_be_built_and_cached(self, empty_cache):
        equation = _equation("bolton", "water")
        objective = get_dewpoint_objective_function(equation)
        assert empty_cache[(type(equation), SurfaceType.WATER)] is objective
        assert objective(293.15, 0.0) > 0
