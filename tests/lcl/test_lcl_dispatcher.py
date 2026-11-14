"""
Tests for Lcl.get_lcl — unified dispatcher
===========================================

``get_lcl`` is the top-level entry point that routes to either
``get_lcl_using_approximation`` or ``get_lcl_using_solver`` based on
``calculation_method``.  These tests focus on routing correctness,
default-argument behaviour, warning emission, and result consistency
with the two underlying path methods.

Physics accuracy for each path is covered in detail in the path-specific
files; tolerances here are kept tight (< 1e-10) because the dispatcher
must be bit-identical to the direct path calls.

Test sections
-------------
1.  Routing — correct method dispatched for each calculation_method value
2.  Default arguments — omitted kwargs produce the same result as explicit ones
3.  Warning behaviour — bolton + solver emits UserWarning and falls back
4.  Return types — scalar → float, array → ndarray of float64
5.  Physical constraints — T_LCL ≤ T_surface, P_LCL ≤ P_surface, Poisson
6.  Bit-identical consistency — get_lcl == direct path call for both strategies
7.  MetPy cross-validation — approximation and solver paths vs Romps (2017)
8.  Enum / string API — calculation_method accepts string and enum
9.  Monotonicity — depression sweep, surface temperature sweep
10. Edge cases
"""

import math
import warnings

import numpy as np
import pytest

from meteocalc.lcl._enums import CalculationMethod, LclEquationName
from meteocalc.lcl.core import Lcl
from meteocalc.shared.constants import Rd, cpd, eps
from meteocalc.vapor.core import Vapor

# ---------------------------------------------------------------------------
# ---- MetPy skip guard -------------------------------------------------------
# ---------------------------------------------------------------------------

metpy = pytest.importorskip("metpy", reason="MetPy not installed — skipping MetPy tests")


def _metpy_lcl(pressure_hpa: float, temp_k: float, dewpoint_k: float):
    """Return (T_LCL_K, P_LCL_hPa) from MetPy's Romps (2017) implementation."""
    from metpy.calc import lcl
    from metpy.units import units

    lcl_p, lcl_t = lcl(
        pressure_hpa * units.hPa,
        temp_k       * units.kelvin,
        dewpoint_k   * units.kelvin,
    )
    return (
        float(lcl_t.to("kelvin").magnitude),
        float(lcl_p.to("hPa").magnitude),
    )


# ---------------------------------------------------------------------------
# ---- Helpers ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _w(dewpoint_k: float, pressure_hpa: float, vapor_eq=None) -> float:
    if vapor_eq is None:
        vapor_eq = Vapor.get_equation("goff_gratch", phase="water")
    e = float(vapor_eq.calculate(dewpoint_k))
    return eps * e / (pressure_hpa - e)


def _w_arr(dewpoint_arr, pressure_arr, vapor_eq=None):
    if vapor_eq is None:
        vapor_eq = Vapor.get_equation("goff_gratch", phase="water")
    e_arr = vapor_eq.calculate(dewpoint_arr).astype(np.float64)
    return eps * e_arr / (pressure_arr - e_arr)


def _lcl(temp_k, dewpoint_k, pressure_hpa, mixing_ratio=None, **kwargs):
    """Wrapper — auto-computes mixing_ratio when not supplied."""
    if mixing_ratio is None and kwargs.get("calculation_method") == "solver":
        mixing_ratio = _w(dewpoint_k, pressure_hpa)
    return Lcl.get_lcl(
        temp_k=temp_k,
        dewpoint_temp_k=dewpoint_k,
        pressure_hpa=pressure_hpa,
        mixing_ratio=mixing_ratio,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ---- Test data --------------------------------------------------------------
# ---------------------------------------------------------------------------

# (label, pressure_hpa, temp_k, dewpoint_k)
STANDARD_CASES = [
    ("standard_midlat",    1013.25, 293.15, 285.15),
    ("warm_humid",         1000.0,  303.15, 298.15),
    ("cool_dry",           1013.25, 278.15, 268.15),
    ("near_saturated",     1013.25, 290.15, 289.15),
    ("high_elevation",     850.0,   285.15, 278.15),
    ("summer_continental", 1000.0,  300.15, 290.15),
    ("autumn_uk",          1013.25, 283.15, 278.15),
    ("moderate_dry",       1013.25, 295.15, 282.15),
]

DEPRESSION_SWEEP = [
    (273.15 + 15, [1, 2, 5, 10, 15]),
    (273.15 + 25, [1, 2, 5, 10, 15]),
]

# Bolton tiered tolerances (for MetPy comparisons on the approximation path)
def _bolton_tol(depression: float) -> float:
    if depression < 3.0:   return 0.3
    elif depression < 10.0: return 0.8
    elif depression < 20.0: return 1.5
    return 2.5


# ===========================================================================
# Section 1 — Routing
# ===========================================================================

class TestRouting:
    """
    ``get_lcl`` must route to the correct underlying method for each
    ``calculation_method`` value.  Verified by checking that the result is
    bit-identical to a direct call on the target method.
    """

    def test_default_routes_to_approximation(self):
        """Omitting ``calculation_method`` must equal explicit ``'approximation'``."""
        t_default, p_default = Lcl.get_lcl(293.15, 285.15, 1013.25)
        t_approx,  p_approx  = Lcl.get_lcl(
            293.15, 285.15, 1013.25, calculation_method="approximation"
        )
        assert t_default == t_approx
        assert p_default == p_approx

    def test_approximation_result_matches_direct_path(self):
        t_disp, p_disp = Lcl.get_lcl(
            293.15, 285.15, 1013.25, calculation_method="approximation"
        )
        t_direct, p_direct = Lcl.get_lcl_using_approximation(
            temp_k=293.15, dewpoint_temp_k=285.15, pressure_hpa=1013.25
        )
        assert t_disp == t_direct
        assert p_disp == p_direct

    def test_solver_result_matches_direct_path(self):
        w = _w(285.15, 1013.25)
        t_disp, p_disp = Lcl.get_lcl(
            293.15, 285.15, 1013.25,
            mixing_ratio=w,
            lcl_equation_name="iterative",
            calculation_method="solver",
        )
        t_direct, p_direct = Lcl.get_lcl_using_solver(
            temp_k=293.15, dewpoint_temp_k=285.15, pressure_hpa=1013.25,
            mixing_ratio=w,
        )
        assert t_disp == t_direct
        assert p_disp == p_direct

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_approximation_path_matches_direct_all_cases(self, label, pressure, temp_k, dewpoint_k):
        t_d, p_d = Lcl.get_lcl(temp_k, dewpoint_k, pressure, calculation_method="approximation")
        t_r, p_r = Lcl.get_lcl_using_approximation(temp_k, dewpoint_k, pressure)
        assert t_d == t_r, f"[{label}] T: dispatcher={t_d:.8f}  direct={t_r:.8f}"
        assert p_d == p_r, f"[{label}] P: dispatcher={p_d:.8f}  direct={p_r:.8f}"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_solver_path_matches_direct_all_cases(self, label, pressure, temp_k, dewpoint_k):
        w = _w(dewpoint_k, pressure)
        t_d, p_d = Lcl.get_lcl(
            temp_k, dewpoint_k, pressure,
            mixing_ratio=w, lcl_equation_name="iterative",
            calculation_method="solver",
        )
        t_r, p_r = Lcl.get_lcl_using_solver(temp_k, dewpoint_k, pressure, mixing_ratio=w)
        assert t_d == t_r, f"[{label}] T: dispatcher={t_d:.8f}  direct={t_r:.8f}"
        assert p_d == p_r, f"[{label}] P: dispatcher={p_d:.8f}  direct={p_r:.8f}"


# ===========================================================================
# Section 2 — Default arguments
# ===========================================================================

class TestDefaultArguments:

    def test_default_calculation_method_is_approximation(self):
        t1, p1 = Lcl.get_lcl(293.15, 285.15, 1013.25)
        t2, p2 = Lcl.get_lcl(293.15, 285.15, 1013.25, calculation_method="approximation")
        assert t1 == t2 and p1 == p2

    def test_default_lcl_equation_name_is_bolton(self):
        t1, p1 = Lcl.get_lcl(293.15, 285.15, 1013.25)
        t2, p2 = Lcl.get_lcl(293.15, 285.15, 1013.25, lcl_equation_name="bolton")
        assert t1 == t2 and p1 == p2

    def test_default_vapor_equation_is_goff_gratch_for_solver(self):
        w = _w(285.15, 1013.25)
        t1, p1 = Lcl.get_lcl(
            293.15, 285.15, 1013.25, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        t2, p2 = Lcl.get_lcl(
            293.15, 285.15, 1013.25, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
            vapor_equation_name="goff_gratch",
        )
        assert t1 == t2 and p1 == p2

    def test_default_surface_type_is_automatic_for_solver(self):
        w = _w(285.15, 1013.25)
        t1, p1 = Lcl.get_lcl(
            293.15, 285.15, 1013.25, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        t2, p2 = Lcl.get_lcl(
            293.15, 285.15, 1013.25, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
            surface_type="automatic",
        )
        assert t1 == t2 and p1 == p2


# ===========================================================================
# Section 3 — Warning behaviour
# ===========================================================================

class TestWarningBehaviour:
    """
    Passing ``lcl_equation_name='bolton'`` with ``calculation_method='solver'``
    must emit a ``UserWarning`` and fall back to the iterative solver.
    """

    def test_bolton_with_solver_emits_userwarning(self):
        w = _w(285.15, 1013.25)
        with pytest.warns(UserWarning, match="bolton"):
            Lcl.get_lcl(
                293.15, 285.15, 1013.25,
                mixing_ratio=w,
                lcl_equation_name="bolton",
                calculation_method="solver",
            )

    def test_bolton_with_solver_falls_back_to_iterative(self):
        """After the warning, result must equal explicit iterative call."""
        w = _w(285.15, 1013.25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            t_warn, p_warn = Lcl.get_lcl(
                293.15, 285.15, 1013.25,
                mixing_ratio=w,
                lcl_equation_name="bolton",
                calculation_method="solver",
            )
        t_iter, p_iter = Lcl.get_lcl_using_solver(
            293.15, 285.15, 1013.25, mixing_ratio=w, solver_name="iterative"
        )
        assert t_warn == t_iter
        assert p_warn == p_iter

    def test_iterative_with_solver_no_warning(self):
        """Explicit ``'iterative'`` + solver must not emit any warning."""
        w = _w(285.15, 1013.25)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Lcl.get_lcl(
                293.15, 285.15, 1013.25,
                mixing_ratio=w,
                lcl_equation_name="iterative",
                calculation_method="solver",
            )

    def test_bolton_with_approximation_no_warning(self):
        """``'bolton'`` + approximation is valid — must not emit any warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Lcl.get_lcl(293.15, 285.15, 1013.25, lcl_equation_name="bolton")


# ===========================================================================
# Section 4 — Return types
# ===========================================================================

class TestReturnTypes:

    def test_scalar_returns_floats(self):
        t, p = Lcl.get_lcl(293.15, 285.15, 1013.25)
        assert isinstance(t, float)
        assert isinstance(p, float)

    def test_returns_two_values(self):
        assert len(Lcl.get_lcl(293.15, 285.15, 1013.25)) == 2

    def test_array_returns_ndarrays(self):
        n    = 5
        t, p = Lcl.get_lcl(
            np.full(n, 293.15), np.full(n, 285.15), np.full(n, 1013.25)
        )
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)

    def test_array_dtype_float64(self):
        n    = 4
        t, p = Lcl.get_lcl(
            np.full(n, 293.15), np.full(n, 285.15), np.full(n, 1013.25)
        )
        assert t.dtype == np.float64
        assert p.dtype == np.float64

    def test_array_shape_matches_input(self):
        n    = 7
        t, p = Lcl.get_lcl(
            np.full(n, 293.15), np.full(n, 285.15), np.full(n, 1013.25)
        )
        assert t.shape == (n,)
        assert p.shape == (n,)

    def test_scalar_array_consistency(self):
        t_sc, p_sc = Lcl.get_lcl(293.15, 285.15, 1013.25)
        t_ar, p_ar = Lcl.get_lcl(
            np.array([293.15]), np.array([285.15]), np.array([1013.25])
        )
        assert abs(t_sc - float(t_ar[0])) < 1e-10
        assert abs(p_sc - float(p_ar[0])) < 1e-10

    def test_solver_path_array_returns_ndarrays(self):
        n      = 4
        t_a    = np.full(n, 293.15)
        td_a   = np.full(n, 285.15)
        p_a    = np.full(n, 1013.25)
        w_a    = _w_arr(td_a, p_a)
        t, p   = Lcl.get_lcl(
            t_a, td_a, p_a, mixing_ratio=w_a,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)
        assert t.shape == (n,)


# ===========================================================================
# Section 5 — Physical constraints
# ===========================================================================

class TestPhysicalConstraints:

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_approx_lcl_temp_below_surface(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, _ = Lcl.get_lcl(temp_k, dewpoint_k, pressure)
        assert t_lcl <= temp_k + 1e-9, f"[{label}] T_LCL={t_lcl:.4f} K > T_surface"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_approx_lcl_pressure_below_surface(self, label, pressure, temp_k, dewpoint_k):
        _, p_lcl = Lcl.get_lcl(temp_k, dewpoint_k, pressure)
        assert p_lcl <= pressure + 1e-9, f"[{label}] P_LCL={p_lcl:.4f} > P_surface"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_approx_poisson_relation(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, p_lcl = Lcl.get_lcl(temp_k, dewpoint_k, pressure)
        expected_p   = pressure * (t_lcl / temp_k) ** (cpd / Rd)
        assert abs(p_lcl - expected_p) < 1e-6, (
            f"[{label}] Poisson: got {p_lcl:.6f} hPa, expected {expected_p:.6f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_solver_lcl_temp_below_surface(self, label, pressure, temp_k, dewpoint_k):
        w = _w(dewpoint_k, pressure)
        t_lcl, _ = Lcl.get_lcl(
            temp_k, dewpoint_k, pressure, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        assert t_lcl <= temp_k + 1e-9, f"[{label}] Solver T_LCL={t_lcl:.4f} K > T_surface"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_results_are_finite(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, p_lcl = Lcl.get_lcl(temp_k, dewpoint_k, pressure)
        assert math.isfinite(t_lcl)
        assert math.isfinite(p_lcl)


# ===========================================================================
# Section 6 — Bit-identical consistency (array inputs)
# ===========================================================================

class TestBitIdenticalConsistency:
    """
    ``get_lcl`` must produce results identical to the direct path methods
    for array inputs as well as scalars.
    """

    def test_approx_array_matches_direct(self):
        temp_arr     = np.array([c[2] for c in STANDARD_CASES])
        dewpoint_arr = np.array([c[3] for c in STANDARD_CASES])
        pressure_arr = np.array([c[1] for c in STANDARD_CASES])

        t_d, p_d = Lcl.get_lcl(temp_arr, dewpoint_arr, pressure_arr)
        t_r, p_r = Lcl.get_lcl_using_approximation(temp_arr, dewpoint_arr, pressure_arr)

        np.testing.assert_array_equal(t_d, t_r)
        np.testing.assert_array_equal(p_d, p_r)

    def test_solver_array_matches_direct(self):
        temp_arr     = np.array([c[2] for c in STANDARD_CASES])
        dewpoint_arr = np.array([c[3] for c in STANDARD_CASES])
        pressure_arr = np.array([c[1] for c in STANDARD_CASES])
        w_arr        = _w_arr(dewpoint_arr, pressure_arr)

        t_d, p_d = Lcl.get_lcl(
            temp_arr, dewpoint_arr, pressure_arr, mixing_ratio=w_arr,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        t_r, p_r = Lcl.get_lcl_using_solver(temp_arr, dewpoint_arr, pressure_arr, w_arr)

        np.testing.assert_array_equal(t_d, t_r)
        np.testing.assert_array_equal(p_d, p_r)


# ===========================================================================
# Section 7 — MetPy cross-validation
# ===========================================================================

class TestMetPyCrossValidation:

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_approximation_path_vs_metpy(self, label, pressure, temp_k, dewpoint_k):
        depression = temp_k - dewpoint_k
        tol        = _bolton_tol(depression)
        t_lcl, _   = Lcl.get_lcl(temp_k, dewpoint_k, pressure)
        t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff       = abs(t_lcl - t_metpy)
        assert diff < tol, (
            f"[{label}] approx: got {t_lcl:.4f} K  MetPy {t_metpy:.4f} K  "
            f"Δ={diff:.4f} K  dep={depression:.1f} K  tol={tol} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_solver_path_vs_metpy(self, label, pressure, temp_k, dewpoint_k):
        w          = _w(dewpoint_k, pressure)
        t_lcl, _   = Lcl.get_lcl(
            temp_k, dewpoint_k, pressure, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff       = abs(t_lcl - t_metpy)
        assert diff < 0.5, (
            f"[{label}] solver: got {t_lcl:.4f} K  MetPy {t_metpy:.4f} K  Δ={diff:.4f} K"
        )

    def test_both_paths_within_tolerance_of_each_other(self):
        """Approximation and solver must agree within Bolton's stated error."""
        for label, pressure, temp_k, dewpoint_k in STANDARD_CASES:
            depression = temp_k - dewpoint_k
            tol        = _bolton_tol(depression)
            w          = _w(dewpoint_k, pressure)

            t_approx, _ = Lcl.get_lcl(temp_k, dewpoint_k, pressure)
            t_solver, _ = Lcl.get_lcl(
                temp_k, dewpoint_k, pressure, mixing_ratio=w,
                lcl_equation_name="iterative", calculation_method="solver",
            )
            diff = abs(t_approx - t_solver)
            assert diff < tol, (
                f"[{label}] approx={t_approx:.4f} K  solver={t_solver:.4f} K  "
                f"Δ={diff:.4f} K  tol={tol} K"
            )


# ===========================================================================
# Section 8 — Enum / string API for calculation_method
# ===========================================================================

class TestEnumStringAPI:

    def test_approximation_string_equals_enum(self):
        t_s, p_s = Lcl.get_lcl(293.15, 285.15, 1013.25, calculation_method="approximation")
        t_e, p_e = Lcl.get_lcl(293.15, 285.15, 1013.25, calculation_method=CalculationMethod.APPROXIMATION)
        assert t_s == t_e and p_s == p_e

    def test_solver_string_equals_enum(self):
        w = _w(285.15, 1013.25)
        t_s, p_s = Lcl.get_lcl(
            293.15, 285.15, 1013.25, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        t_e, p_e = Lcl.get_lcl(
            293.15, 285.15, 1013.25, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method=CalculationMethod.SOLVER,
        )
        assert t_s == t_e and p_s == p_e

    def test_invalid_calculation_method_raises(self):
        with pytest.raises((ValueError, KeyError)):
            Lcl.get_lcl(293.15, 285.15, 1013.25, calculation_method="nonexistent")


# ===========================================================================
# Section 9 — Monotonicity
# ===========================================================================

class TestMonotonicity:

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_approx_larger_depression_lowers_lcl_temp(self, base_temp, depressions):
        t_prev = None
        for dep in sorted(depressions):
            t_lcl, _ = Lcl.get_lcl(base_temp, base_temp - dep, 1013.25)
            if t_prev is not None:
                assert t_lcl < t_prev
            t_prev = t_lcl

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_solver_larger_depression_lowers_lcl_temp(self, base_temp, depressions):
        t_prev = None
        for dep in sorted(depressions):
            td = base_temp - dep
            w  = _w(td, 1013.25)
            t_lcl, _ = Lcl.get_lcl(
                base_temp, td, 1013.25, mixing_ratio=w,
                lcl_equation_name="iterative", calculation_method="solver",
            )
            if t_prev is not None:
                assert t_lcl < t_prev
            t_prev = t_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_approx_warmer_parcel_warmer_lcl(self, depression):
        temps  = [278.15, 283.15, 288.15, 293.15, 298.15]
        t_prev = None
        for t in temps:
            t_lcl, _ = Lcl.get_lcl(t, t - depression, 1013.25)
            if t_prev is not None:
                assert t_lcl > t_prev
            t_prev = t_lcl


# ===========================================================================
# Section 10 — Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_very_small_depression_approximation(self):
        t_lcl, p_lcl = Lcl.get_lcl(293.15, 292.65, 1013.25)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 293.15

    def test_very_small_depression_solver(self):
        w = _w(292.65, 1013.25)
        t_lcl, p_lcl = Lcl.get_lcl(
            293.15, 292.65, 1013.25, mixing_ratio=w,
            lcl_equation_name="iterative", calculation_method="solver",
        )
        assert math.isfinite(t_lcl)
        assert t_lcl <= 293.15

    def test_high_surface_pressure(self):
        t_lcl, p_lcl = Lcl.get_lcl(293.15, 285.15, 1050.0)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 1050.0

    def test_high_elevation_surface(self):
        t_lcl, p_lcl = Lcl.get_lcl(285.15, 278.15, 850.0)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 850.0

    def test_deterministic_approximation(self):
        results = [Lcl.get_lcl(293.15, 285.15, 1013.25) for _ in range(5)]
        for t, p in results:
            assert t == results[0][0] and p == results[0][1]

    def test_deterministic_solver(self):
        w = _w(285.15, 1013.25)
        results = [
            Lcl.get_lcl(
                293.15, 285.15, 1013.25, mixing_ratio=w,
                lcl_equation_name="iterative", calculation_method="solver",
            )
            for _ in range(5)
        ]
        for t, p in results:
            assert t == results[0][0] and p == results[0][1]

    def test_large_random_array_approximation(self):
        rng          = np.random.default_rng(42)
        n            = 10_000
        temp_k       = rng.uniform(280.0, 308.0, n)
        dewpoint_k   = temp_k - rng.uniform(1.0, 10.0, n)
        pressure_hpa = rng.uniform(850.0, 1013.25, n)
        t_lcl, p_lcl = Lcl.get_lcl(temp_k, dewpoint_k, pressure_hpa)
        assert np.all(np.isfinite(t_lcl))
        assert np.all(np.isfinite(p_lcl))
        assert np.all(t_lcl <= temp_k + 1e-9)
        assert np.all(p_lcl <= pressure_hpa + 1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])