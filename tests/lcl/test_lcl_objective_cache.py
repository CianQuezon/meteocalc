"""
Tests for the compiled objective-function cache in the LCL solver.

``meteocalc.lcl._solver_method`` keeps one ``@njit`` objective per vapor
equation class in ``_OBJECTIVE_CACHE``. The objective embeds both the water and
the ice constants and picks one at run time from its ``below_freezing``
argument, so one compiled function serves every surface type. Building a fresh
closure per call would give Numba a new function type and force rapid-roots to
compile a new solver specialisation every time. These tests check that:

1. The cache key is the equation class: instances and surface types share one
   entry, different classes do not.
2. A cached objective evaluates the LCL residual exactly, with the right
   constants for each ``below_freezing`` value, whichever instance built it.
3. Solver results are identical whether the cache is cold, warm, bypassed, or
   shared between interleaved equation classes.
4. A warm call compiles nothing, and rapid-roots does not accumulate solver
   specialisations across repeated calls.

Tests that need an empty cache swap in a fresh dict with ``monkeypatch`` so the
module-level cache is restored afterwards and other tests are not affected.
"""

import warnings

import numpy as np
import pytest
from numba.core import event as numba_event
from numpy.testing import assert_allclose, assert_array_equal

import meteocalc.lcl._solver_method as solver_module
from meteocalc.lcl import Lcl
from meteocalc.lcl._solver_method import (
    _get_lcl_objective_function,
    get_lcl_using_solver,
)
from meteocalc.shared._shared_enums import SurfaceType
from meteocalc.shared.constants import Rd, cpd, eps
from meteocalc.vapor._jit_equations import (
    _bolton_scalar,
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
    BoltonEquation,
    GoffGratchEquation,
    HylandWexlerEquation,
)

# ==============================================================================
# Inputs and helpers
# ==============================================================================

# Independent reference: the JIT function and the (water, ice) constants each
# equation class must use, taken straight from the vapor module. Bolton has no
# constants and no ice form, so both branches use the same formula.
REFERENCE = {
    GoffGratchEquation: (_goff_gratch_scalar, GOFF_GRATCH_WATER, GOFF_GRATCH_ICE),
    HylandWexlerEquation: (
        _hyland_wexler_scalar,
        HYLAND_WEXLER_WATER,
        HYLAND_WEXLER_ICE,
    ),
    BoltonEquation: (_bolton_scalar, (), ()),
}

EQUATION_CLASSES = [GoffGratchEquation, HylandWexlerEquation, BoltonEquation]
ICE_CAPABLE_CLASSES = [GoffGratchEquation, HylandWexlerEquation]

# Bolton has no ice constants, so building its objective warns once.
ignore_bolton_ice_warning = pytest.mark.filterwarnings(
    "ignore:Bolton does not support ice surface constants"
)

N = 64

# Warm parcels: surface above 0 °C, so the solver uses the water branch.
WARM_TEMPS = np.linspace(280.15, 305.15, N)
WARM_DEWPOINTS = WARM_TEMPS - np.linspace(15.0, 2.0, N)
WARM_PRESSURES = np.linspace(900.0, 1010.0, N)

# Cold parcels: surface below 0 °C, so the solver uses the ice branch.
COLD_TEMPS = np.linspace(245.15, 270.15, N)
COLD_DEWPOINTS = COLD_TEMPS - np.linspace(12.0, 3.0, N)
COLD_PRESSURES = np.linspace(700.0, 1000.0, N)


def _reference_e_sat(equation_class, temp_k, below_freezing):
    scalar_func, water, ice = REFERENCE[equation_class]
    constants = ice if below_freezing else water
    return scalar_func(float(temp_k), *constants)


def _reference_objective(equation_class, x, w, p, t, below_freezing):
    p_lcl = p * (x / t) ** (cpd / Rd)
    e_sat = _reference_e_sat(equation_class, x, below_freezing)
    return eps * e_sat / (p_lcl - e_sat) - w


def _mixing_ratio(equation_class, dewpoints, pressures, below_freezing):
    """Surface mixing ratio from the same formula the solver will use."""
    e = np.array(
        [_reference_e_sat(equation_class, td, below_freezing) for td in dewpoints]
    )
    return eps * e / (pressures - e)


def _parcels(equation_class, cold=False):
    if cold:
        temps, dewpoints, pressures = COLD_TEMPS, COLD_DEWPOINTS, COLD_PRESSURES
    else:
        temps, dewpoints, pressures = WARM_TEMPS, WARM_DEWPOINTS, WARM_PRESSURES
    w = _mixing_ratio(equation_class, dewpoints, pressures, below_freezing=cold)
    return temps, dewpoints, pressures, w


def _solve(equation, cold=False):
    temps, dewpoints, pressures, w = _parcels(type(equation), cold)
    return get_lcl_using_solver(
        temp_k=temps,
        dewpoint_k=dewpoints,
        pressure_hpa=pressures,
        mixing_ratio=w,
        vapor_equation=equation,
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

    def test_first_call_stores_one_entry_under_the_class(self, empty_cache):
        objective = _get_lcl_objective_function(GoffGratchEquation("water"))

        assert list(empty_cache) == [GoffGratchEquation]
        assert empty_cache[GoffGratchEquation] is objective

    def test_objective_is_a_numba_dispatcher(self, empty_cache):
        objective = _get_lcl_objective_function(GoffGratchEquation("water"))
        # Only a Numba dispatcher can be passed into the compiled solver; a
        # plain Python function would mean the cache stores the wrong thing.
        assert hasattr(objective, "overloads")
        assert hasattr(objective, "py_func")

    def test_same_instance_returns_same_object(self, empty_cache):
        equation = GoffGratchEquation("water")
        assert _get_lcl_objective_function(equation) is _get_lcl_objective_function(
            equation
        )

    def test_new_instances_of_same_class_share_objective(self, empty_cache):
        # The public API builds a new equation instance on every call, so the
        # cache only helps if it is keyed by class rather than by instance.
        first = _get_lcl_objective_function(GoffGratchEquation("water"))
        second = _get_lcl_objective_function(GoffGratchEquation("water"))
        assert first is second
        assert len(empty_cache) == 1

    @pytest.mark.parametrize("equation_class", ICE_CAPABLE_CLASSES)
    def test_surface_types_share_one_objective(self, empty_cache, equation_class):
        # By design: the objective holds both constant sets and selects at run
        # time, so the surface type is not part of the key.
        objectives = {
            surface: _get_lcl_objective_function(equation_class(surface))
            for surface in ("water", "ice", "automatic")
        }
        assert objectives["water"] is objectives["ice"] is objectives["automatic"]
        assert list(empty_cache) == [equation_class]

    @ignore_bolton_ice_warning
    def test_equation_classes_get_separate_objectives(self, empty_cache):
        objectives = [
            _get_lcl_objective_function(equation_class("water"))
            for equation_class in EQUATION_CLASSES
        ]
        assert len({id(objective) for objective in objectives}) == 3
        assert set(empty_cache) == set(EQUATION_CLASSES)

    @ignore_bolton_ice_warning
    def test_repeated_calls_do_not_grow_the_cache(self, empty_cache):
        for _ in range(25):
            for equation_class in EQUATION_CLASSES:
                _get_lcl_objective_function(equation_class("water"))
            for equation_class in ICE_CAPABLE_CLASSES:
                _get_lcl_objective_function(equation_class("ice"))
        assert len(empty_cache) == len(EQUATION_CLASSES)

    def test_cache_hit_does_not_rebuild(self, empty_cache, monkeypatch):
        calls = []
        original = GoffGratchEquation.get_constants

        def counting_get_constants(self, surface_type=None):
            calls.append(surface_type)
            return original(self, surface_type)

        monkeypatch.setattr(GoffGratchEquation, "get_constants", counting_get_constants)

        for surface in ("water", "ice", "automatic") * 5:
            _get_lcl_objective_function(GoffGratchEquation(surface))

        # One build reads the ice and the water constants once each.
        assert set(calls) == {SurfaceType.ICE, SurfaceType.WATER}
        assert len(calls) == 2

    def test_bolton_ice_warning_is_raised_only_when_building(self, empty_cache):
        with pytest.warns(UserWarning, match="Bolton does not support ice"):
            _get_lcl_objective_function(BoltonEquation("water"))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _get_lcl_objective_function(BoltonEquation("water"))

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
            _get_lcl_objective_function(GoffGratchEquation("water"))
        assert empty_cache == {}

        state["fail"] = False
        objective = _get_lcl_objective_function(GoffGratchEquation("water"))
        assert empty_cache[GoffGratchEquation] is objective


# ==============================================================================
# 2. Values computed by cached objectives
# ==============================================================================


class TestCachedObjectiveValues:
    """A cached objective must compute rs(x, p_LCL) - w exactly."""

    @ignore_bolton_ice_warning
    @pytest.mark.parametrize("equation_class", EQUATION_CLASSES)
    @pytest.mark.parametrize("below_freezing", [0.0, 1.0])
    def test_objective_matches_reference_formula(self, equation_class, below_freezing):
        objective = _get_lcl_objective_function(equation_class("water"))

        temps = COLD_TEMPS if below_freezing else WARM_TEMPS
        for t in temps[::8]:
            for x in (t - 12.0, t - 4.0, t):
                args = (float(x), 0.004, 950.0, float(t))
                expected = _reference_objective(equation_class, *args, below_freezing)
                assert_allclose(objective(*args, below_freezing), expected, rtol=1e-12)

    @pytest.mark.parametrize("equation_class", ICE_CAPABLE_CLASSES)
    @pytest.mark.parametrize("builder", ["water", "ice", "automatic"])
    def test_both_branches_are_correct_whichever_instance_built_it(
        self, empty_cache, equation_class, builder
    ):
        # The cache is keyed by class only, so the instance that happens to
        # build the objective must not leave its surface type baked in.
        objective = _get_lcl_objective_function(equation_class(builder))
        args = (258.15, 0.002, 850.0, 263.15)

        water = objective(*args, 0.0)
        ice = objective(*args, 1.0)

        assert_allclose(
            water, _reference_objective(equation_class, *args, False), rtol=1e-12
        )
        assert_allclose(
            ice, _reference_objective(equation_class, *args, True), rtol=1e-12
        )
        assert water != ice

    @ignore_bolton_ice_warning
    def test_bolton_branches_are_identical(self):
        objective = _get_lcl_objective_function(BoltonEquation("water"))
        args = (283.15, 0.006, 950.0, 293.15)
        assert objective(*args, 0.0) == objective(*args, 1.0)

    def test_equation_classes_compute_different_values(self):
        args = (283.15, 0.0, 950.0, 293.15, 0.0)
        goff_gratch = _get_lcl_objective_function(GoffGratchEquation("water"))(*args)
        hyland_wexler = _get_lcl_objective_function(HylandWexlerEquation("water"))(
            *args
        )
        # Close but not identical; identical output would mean one class was
        # served the other's objective.
        assert goff_gratch != hyland_wexler
        assert_allclose(goff_gratch, hyland_wexler, rtol=5e-3)


# ==============================================================================
# 3. Solver results with the cache
# ==============================================================================


class TestSolverResultsWithCache:
    """The cache must not change any result, bit for bit."""

    @pytest.mark.parametrize("cold", [False, True], ids=["warm_air", "cold_air"])
    def test_cold_and_warm_cache_give_identical_results(self, empty_cache, cold):
        first = _solve(GoffGratchEquation("water"), cold)
        second = _solve(GoffGratchEquation("water"), cold)

        assert first[2].all()
        for first_part, second_part in zip(first, second):
            assert_array_equal(second_part, first_part)

    @ignore_bolton_ice_warning
    @pytest.mark.parametrize("equation_class", EQUATION_CLASSES)
    def test_results_are_physically_valid(self, equation_class):
        temps, dewpoints, pressures, w = _parcels(equation_class)
        roots, _, converged = _solve(equation_class("water"))

        assert converged.all()
        assert np.all(roots < dewpoints)

        # At the root the parcel's saturation mixing ratio equals w. Brent stops
        # at a 1e-6 K bracket, about 1e-7 relative in the mixing ratio.
        for root, t, p, w_i in zip(roots, temps, pressures, w):
            residual = _reference_objective(equation_class, root, w_i, p, t, False)
            assert abs(residual) <= 1e-6 * w_i

    @pytest.mark.parametrize("equation_class", ICE_CAPABLE_CLASSES)
    def test_cold_air_results_satisfy_the_ice_formula(self, equation_class):
        temps, dewpoints, pressures, w = _parcels(equation_class, cold=True)
        roots, _, converged = _solve(equation_class("ice"), cold=True)

        assert converged.all()
        for root, t, p, w_i in zip(roots, temps, pressures, w):
            residual = _reference_objective(equation_class, root, w_i, p, t, True)
            assert abs(residual) <= 1e-6 * w_i

    def test_instance_that_built_the_cache_does_not_change_results(self, monkeypatch):
        monkeypatch.setattr(solver_module, "_OBJECTIVE_CACHE", {})
        built_by_water = _solve(GoffGratchEquation("water"))

        monkeypatch.setattr(solver_module, "_OBJECTIVE_CACHE", {})
        _get_lcl_objective_function(GoffGratchEquation("ice"))
        built_by_ice = _solve(GoffGratchEquation("water"))

        for water_part, ice_part in zip(built_by_water, built_by_ice):
            assert_array_equal(ice_part, water_part)

    @ignore_bolton_ice_warning
    def test_interleaved_equations_match_isolated_runs(self):
        first = {cls: _solve(cls("water"))[0] for cls in EQUATION_CLASSES}
        cold_first = _solve(GoffGratchEquation("ice"), cold=True)[0]
        again = {cls: _solve(cls("water"))[0] for cls in reversed(EQUATION_CLASSES)}
        cold_again = _solve(GoffGratchEquation("ice"), cold=True)[0]

        for equation_class in EQUATION_CLASSES:
            assert_array_equal(again[equation_class], first[equation_class])
        assert_array_equal(cold_again, cold_first)

        assert not np.array_equal(
            first[GoffGratchEquation], first[HylandWexlerEquation]
        )
        assert_allclose(
            first[GoffGratchEquation], first[HylandWexlerEquation], atol=0.05
        )

    def test_scalar_and_array_calls_agree(self):
        temps, dewpoints, pressures, w = _parcels(GoffGratchEquation)
        roots, _, _ = _solve(GoffGratchEquation("water"))

        for i in (0, 31, 63):
            root, _, converged = get_lcl_using_solver(
                temp_k=float(temps[i]),
                dewpoint_k=float(dewpoints[i]),
                pressure_hpa=float(pressures[i]),
                mixing_ratio=float(w[i]),
                vapor_equation=GoffGratchEquation("water"),
            )
            assert isinstance(root, float)
            assert converged
            assert root == roots[i]

    def test_public_api_uses_the_same_cache_entry(self, empty_cache):
        temps, dewpoints, pressures, w = _parcels(GoffGratchEquation)
        lcl_temp, lcl_pressure = Lcl.get_lcl_using_solver(
            temps, dewpoints, pressures, w, surface_type="water"
        )
        cold_temps, cold_dewpoints, cold_pressures, cold_w = _parcels(
            GoffGratchEquation, cold=True
        )
        Lcl.get_lcl_using_solver(
            cold_temps, cold_dewpoints, cold_pressures, cold_w, surface_type="ice"
        )

        assert list(empty_cache) == [GoffGratchEquation]
        assert_array_equal(lcl_temp, _solve(GoffGratchEquation("water"))[0])
        assert_allclose(
            lcl_pressure, pressures * (lcl_temp / temps) ** (cpd / Rd), rtol=1e-12
        )
        assert len(empty_cache) == 1

    @pytest.mark.slow
    def test_cached_results_match_uncached_results(self, monkeypatch):
        cached = _solve(HylandWexlerEquation("ice"), cold=True)

        monkeypatch.setattr(solver_module, "_OBJECTIVE_CACHE", _NoStoreDict())
        uncached = _solve(HylandWexlerEquation("ice"), cold=True)

        for cached_part, uncached_part in zip(cached, uncached):
            assert_array_equal(cached_part, uncached_part)


# ==============================================================================
# 4. Compilation
# ==============================================================================


class TestNoRecompilation:
    """The point of the cache: warm calls must not compile anything."""

    @ignore_bolton_ice_warning
    @pytest.mark.parametrize("equation_class", EQUATION_CLASSES)
    def test_warm_array_call_compiles_nothing(self, equation_class):
        _solve(equation_class("water"))  # warm up
        assert _count_compilations(_solve, equation_class("water")) == 0

    @pytest.mark.parametrize("equation_class", ICE_CAPABLE_CLASSES)
    def test_warm_cold_air_call_compiles_nothing(self, equation_class):
        _solve(equation_class("ice"), cold=True)  # warm up
        assert _count_compilations(_solve, equation_class("ice"), cold=True) == 0

    def test_warm_scalar_call_compiles_nothing(self):
        def scalar_call():
            return get_lcl_using_solver(
                temp_k=293.15,
                dewpoint_k=285.15,
                pressure_hpa=1000.0,
                mixing_ratio=0.0092,
                vapor_equation=GoffGratchEquation("water"),
            )

        scalar_call()  # warm up
        assert _count_compilations(scalar_call) == 0

    def test_warm_public_api_calls_compile_nothing(self):
        temps, dewpoints, pressures, w = _parcels(GoffGratchEquation)

        def calls():
            Lcl.get_lcl_using_solver(temps, dewpoints, pressures, w)
            Lcl.get_lcl_using_solver(
                temps, dewpoints, pressures, w, surface_type="water"
            )
            Lcl.get_lcl_using_solver(293.15, 285.15, 1000.0, 0.0092)

        calls()  # warm up
        assert _count_compilations(calls) == 0

    @ignore_bolton_ice_warning
    def test_solver_specialisations_do_not_grow(self):
        try:
            from rapid_roots.solvers._codegen import generate_vectorised_solver
            from rapid_roots.solvers._jit_solvers import _brent_scalar
        except ImportError:
            pytest.skip("rapid-roots internals have moved")

        for equation_class in EQUATION_CLASSES:  # warm up every class
            _solve(equation_class("water"))

        brent_solver = generate_vectorised_solver(_brent_scalar, 4, "bracket")
        specialisations = len(brent_solver.overloads)
        assert specialisations >= 1  # confirms this is the solver in use

        for _ in range(10):
            for equation_class in EQUATION_CLASSES:
                _solve(equation_class("water"))
            _solve(GoffGratchEquation("ice"), cold=True)

        assert len(brent_solver.overloads) == specialisations

    @pytest.mark.slow
    def test_without_the_cache_every_call_compiles(self, no_store_cache):
        # Negative control: proves the compile counter can see the regression
        # the cache prevents, so a zero above is meaningful.
        _solve(GoffGratchEquation("water"))
        assert _count_compilations(_solve, GoffGratchEquation("water")) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])