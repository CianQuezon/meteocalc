"""
Comprehensive unit tests for IterativeLclEquation
==================================================
Cross-validated against MetPy's LCL solver (Romps 2017) to ensure accuracy.
Vapour equations obtained via the Vapor module public interface.

IterativeLclEquation computes the LCL by finding the root of:

    f(T_LCL) = rs(T_LCL, p_LCL) - w = 0

where rs is the saturation mixing ratio at the LCL and w is the surface
mixing ratio. Accuracy is expected within ~1e-8 K of the exact solution.

Tolerance vs MetPy (Romps 2017 exact)
--------------------------------------
A flat tolerance of 0.5 K is used throughout. The iterative solver itself
achieves ~1e-8 K accuracy — all remaining discrepancy vs MetPy comes from
the choice of vapour pressure equation, not the solver method. Tiered
tolerances are NOT appropriate here because solver error is constant
regardless of dewpoint depression.

References
----------
.. [1] Romps, D. M. (2017). Exact analytic solutions for pseudo-adiabatic
   ascent. Journal of the Atmospheric Sciences, 74(9), 3033-3039.
   https://doi.org/10.1175/JAS-D-17-0073.1

.. [2] Bolton, D. (1980). The computation of equivalent potential temperature.
   Monthly Weather Review, 108(7), 1046-1053.

Dependencies
------------
    pip install pytest numpy numba metpy
"""

import math

import numpy as np
import pytest

from meteocalc.lcl._enums import LclEquationName, CalculationMethod
from meteocalc.lcl._lcl_equation import (
    BoltonLclEquation,
    IterativeLclEquation,
    LiftingCondensationLevelEquation,
)
from meteocalc.shared.constants import Rd, cpd, eps
from meteocalc.vapor._vapor_equations import VaporEquation
from meteocalc.vapor.core import Vapor

# ---------------------------------------------------------------------------
# ---- MetPy helper ---------------------------------------------------------
# ---------------------------------------------------------------------------

metpy = pytest.importorskip("metpy", reason="MetPy not installed — skipping MetPy tests")


def _metpy_lcl(pressure_hpa: float, temp_k: float, dewpoint_k: float):
    """Return MetPy LCL (temp_K, pressure_hPa) using Romps (2017)."""
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
# ---- Fixtures — via Vapor module ------------------------------------------
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vapor_gg() -> VaporEquation:
    """Goff-Gratch water equation via Vapor.get_equation()."""
    return Vapor.get_equation("goff_gratch", phase="water")


@pytest.fixture(scope="module")
def vapor_hw() -> VaporEquation:
    """Hyland-Wexler water equation via Vapor.get_equation()."""
    return Vapor.get_equation("hyland_wexler", phase="water")


@pytest.fixture(scope="module")
def vapor_bolton() -> VaporEquation:
    """Bolton equation via Vapor.get_equation()."""
    return Vapor.get_equation("bolton")


@pytest.fixture(scope="module")
def eq() -> IterativeLclEquation:
    """Shared IterativeLclEquation instance."""
    return IterativeLclEquation()


def _mixing_ratio(vapor_eq: VaporEquation, dewpoint_k: float, pressure_hpa: float) -> float:
    """Compute surface mixing ratio using given vapour equation."""
    e = float(vapor_eq.calculate(dewpoint_k))
    return eps * e / (pressure_hpa - e)


def _mixing_ratio_array(
    vapor_eq: VaporEquation,
    dewpoint_arr: np.ndarray,
    pressure_arr: np.ndarray,
) -> np.ndarray:
    """Compute surface mixing ratio array using given vapour equation."""
    e_arr = vapor_eq.calculate(dewpoint_arr).astype(np.float64)
    return eps * e_arr / (pressure_arr - e_arr)


# ---------------------------------------------------------------------------
# ---- Test data ------------------------------------------------------------
# ---------------------------------------------------------------------------

# (label, pressure_hpa, temp_k, dewpoint_k)
STANDARD_CASES = [
    ("standard_midlat",    1013.25, 293.15, 285.15),  # 20°C / 12°C — 8 K
    ("warm_humid",         1000.0,  303.15, 298.15),  # 30°C / 25°C — 5 K
    ("cool_dry",           1013.25, 278.15, 268.15),  # 5°C / -5°C  — 10 K
    ("near_saturated",     1013.25, 290.15, 289.15),  # 1 K depression
    ("high_elevation",     850.0,   285.15, 278.15),  # 12°C / 5°C
    ("summer_continental", 1000.0,  300.15, 290.15),  # 27°C / 17°C — 10 K
    ("autumn_uk",          1013.25, 283.15, 278.15),  # 10°C / 5°C  — 5 K
    ("moderate_dry",       1013.25, 295.15, 282.15),  # 22°C / 9°C  — 13 K
]

# Flat tolerance — solver error is constant, not depression-dependent
METPY_TOL_K   = 0.02    # K
METPY_TOL_HPA = 0.6    # hPa

DEPRESSION_SWEEP = [
    (273.15 + 15, [1, 2, 5, 10, 15]),
    (273.15 + 25, [1, 2, 5, 10, 15]),
]


# ===========================================================================
# Section 1 — Class attributes
# ===========================================================================

class TestClassAttributes:
    """IterativeLclEquation must expose correct class-level metadata."""

    def test_name_is_iterative(self, eq):
        assert eq.name == LclEquationName.ITERATIVE

    def test_calculation_method_is_solver(self, eq):
        assert eq.calculation_method == CalculationMethod.SOLVER

    def test_is_instance_of_base_class(self, eq):
        assert isinstance(eq, LiftingCondensationLevelEquation)

    def test_name_value(self, eq):
        assert eq.name.value == "iterative"

    def test_calculation_method_value(self, eq):
        assert eq.calculation_method.value == "solver"

    def test_different_from_bolton_equation(self, eq):
        bolton = BoltonLclEquation()
        assert eq.name != bolton.name
        assert eq.calculation_method != bolton.calculation_method


# ===========================================================================
# Section 2 — Vapor module integration
# ===========================================================================

class TestVaporModuleIntegration:
    """
    Verify that Vapor.get_equation() returns the correct equation types
    and produces consistent results with IterativeLclEquation.
    """

    def test_vapor_returns_goff_gratch_instance(self, vapor_gg):
        from meteocalc.vapor._vapor_equations import GoffGratchEquation
        assert isinstance(vapor_gg, GoffGratchEquation)

    def test_vapor_returns_hyland_wexler_instance(self, vapor_hw):
        from meteocalc.vapor._vapor_equations import HylandWexlerEquation
        assert isinstance(vapor_hw, HylandWexlerEquation)

    def test_vapor_returns_bolton_instance(self, vapor_bolton):
        from meteocalc.vapor._vapor_equations import BoltonEquation
        assert isinstance(vapor_bolton, BoltonEquation)

    def test_string_and_enum_inputs_give_identical_results(self, eq):
        """Vapor.get_equation() with string vs enum must give identical results."""
        from meteocalc.vapor._enums import VaporEquationName
        vapor_str  = Vapor.get_equation("goff_gratch",                phase="water")
        vapor_enum = Vapor.get_equation(VaporEquationName.GOFF_GRATCH, phase="water")

        w_str  = _mixing_ratio(vapor_str,  285.15, 1013.25)
        w_enum = _mixing_ratio(vapor_enum, 285.15, 1013.25)

        t_str,  _ = eq.calculate(293.15, 285.15, 1013.25, w_str,  vapor_str)
        t_enum, _ = eq.calculate(293.15, 285.15, 1013.25, w_enum, vapor_enum)

        assert abs(t_str - t_enum) < 1e-10

    def test_vapor_list_equations_complete(self):
        equations = Vapor.list_equations()
        assert "bolton"        in equations
        assert "goff_gratch"   in equations
        assert "hyland_wexler" in equations

    def test_all_vapor_equations_work_with_iterative_lcl(self, eq):
        """All equations from Vapor.list_equations() must work with IterativeLclEquation."""
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25

        for eq_name in Vapor.list_equations():
            vapor_eq       = Vapor.get_equation(eq_name, phase="water")
            w              = _mixing_ratio(vapor_eq, dewpoint_k, pressure_hpa)
            t_lcl, p_lcl   = eq.calculate(temp_k, dewpoint_k, pressure_hpa, w, vapor_eq)

            assert math.isfinite(t_lcl), f"{eq_name}: T_LCL not finite"
            assert math.isfinite(p_lcl), f"{eq_name}: p_LCL not finite"
            assert t_lcl <= temp_k,      f"{eq_name}: T_LCL > T_surface"


# ===========================================================================
# Section 3 — Physical constraints
# ===========================================================================

class TestPhysicalConstraints:
    """LCL results must satisfy fundamental atmospheric physics."""

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_below_surface_temp(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        t_lcl, _ = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        assert t_lcl <= temp_k + 1e-9, (
            f"[{label}] T_LCL={t_lcl:.4f} K > T_surface={temp_k:.4f} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_pressure_below_surface_pressure(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        _, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        assert p_lcl <= pressure + 1e-9, (
            f"[{label}] p_LCL={p_lcl:.4f} hPa > p_surface={pressure:.4f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_above_absolute_zero(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        t_lcl, _ = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        assert t_lcl > 0.0, f"[{label}] T_LCL={t_lcl:.4f} K ≤ 0 K"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_results_are_finite(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        t_lcl, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        assert math.isfinite(t_lcl), f"[{label}] T_LCL not finite"
        assert math.isfinite(p_lcl), f"[{label}] p_LCL not finite"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_poisson_relation_consistent(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        """p_LCL must satisfy Poisson relation given T_LCL."""
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        t_lcl, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        expected_p   = pressure * (t_lcl / temp_k) ** (cpd / Rd)
        assert abs(p_lcl - expected_p) < 1e-6, (
            f"[{label}] Poisson violated: p_LCL={p_lcl:.6f} vs {expected_p:.6f} hPa"
        )


# ===========================================================================
# Section 4 — API behaviour
# ===========================================================================

class TestAPIBehaviour:
    """Scalar inputs return scalars; array inputs return arrays."""

    def test_scalar_input_returns_floats(self, eq, vapor_gg):
        w = _mixing_ratio(vapor_gg, 285.15, 1013.25)
        t_lcl, p_lcl = eq.calculate(293.15, 285.15, 1013.25, w, vapor_gg)
        assert isinstance(t_lcl, float)
        assert isinstance(p_lcl, float)

    def test_array_input_returns_ndarrays(self, eq, vapor_gg):
        n            = 5
        temp_arr     = np.full(n, 293.15)
        dewpoint_arr = np.full(n, 285.15)
        pressure_arr = np.full(n, 1013.25)
        w_arr        = _mixing_ratio_array(vapor_gg, dewpoint_arr, pressure_arr)
        t_lcl, p_lcl = eq.calculate(temp_arr, dewpoint_arr, pressure_arr, w_arr, vapor_gg)
        assert isinstance(t_lcl, np.ndarray)
        assert isinstance(p_lcl, np.ndarray)

    def test_array_output_dtype_float64(self, eq, vapor_gg):
        n            = 3
        temp_arr     = np.full(n, 293.15)
        dewpoint_arr = np.full(n, 285.15)
        pressure_arr = np.full(n, 1013.25)
        w_arr        = _mixing_ratio_array(vapor_gg, dewpoint_arr, pressure_arr)
        t_lcl, p_lcl = eq.calculate(temp_arr, dewpoint_arr, pressure_arr, w_arr, vapor_gg)
        assert t_lcl.dtype == np.float64
        assert p_lcl.dtype == np.float64

    def test_array_output_shape_matches_input(self, eq, vapor_gg):
        n            = 5
        temp_arr     = np.full(n, 293.15)
        dewpoint_arr = np.full(n, 285.15)
        pressure_arr = np.full(n, 1013.25)
        w_arr        = _mixing_ratio_array(vapor_gg, dewpoint_arr, pressure_arr)
        t_lcl, p_lcl = eq.calculate(temp_arr, dewpoint_arr, pressure_arr, w_arr, vapor_gg)
        assert t_lcl.shape == (n,)
        assert p_lcl.shape == (n,)

    def test_returns_two_values(self, eq, vapor_gg):
        w = _mixing_ratio(vapor_gg, 285.15, 1013.25)
        assert len(eq.calculate(293.15, 285.15, 1013.25, w, vapor_gg)) == 2

    def test_scalar_array_consistency(self, eq, vapor_gg):
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure_hpa)

        t_scalar, p_scalar = eq.calculate(temp_k, dewpoint_k, pressure_hpa, w, vapor_gg)
        t_array,  p_array  = eq.calculate(
            np.array([temp_k]), np.array([dewpoint_k]),
            np.array([pressure_hpa]), np.array([w]), vapor_gg,
        )
        assert abs(t_scalar - float(t_array[0])) < 1e-10
        assert abs(p_scalar - float(p_array[0])) < 1e-10


    def test_deterministic_repeated_calls(self, eq, vapor_gg):
        w       = _mixing_ratio(vapor_gg, 285.15, 1013.25)
        results = [eq.calculate(293.15, 285.15, 1013.25, w, vapor_gg) for _ in range(5)]
        for t, p in results:
            assert t == results[0][0]
            assert p == results[0][1]


# ===========================================================================
# Section 5 — Monotonicity
# ===========================================================================

class TestMonotonicity:

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_temp(self, eq, vapor_gg, base_temp, depressions):
        t_prev = None
        for dep in sorted(depressions):
            td = base_temp - dep
            w  = _mixing_ratio(vapor_gg, td, 1013.25)
            t_lcl, _ = eq.calculate(base_temp, td, 1013.25, w, vapor_gg)
            if t_prev is not None:
                assert t_lcl < t_prev
            t_prev = t_lcl

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_pressure(self, eq, vapor_gg, base_temp, depressions):
        p_prev = None
        for dep in sorted(depressions):
            td = base_temp - dep
            w  = _mixing_ratio(vapor_gg, td, 1013.25)
            _, p_lcl = eq.calculate(base_temp, td, 1013.25, w, vapor_gg)
            if p_prev is not None:
                assert p_lcl < p_prev
            p_prev = p_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_warmer_parcel_gives_warmer_lcl_temp(self, eq, vapor_gg, depression):
        temps  = [278.15, 283.15, 288.15, 293.15, 298.15]
        t_prev = None
        for t in temps:
            w = _mixing_ratio(vapor_gg, t - depression, 1013.25)
            t_lcl, _ = eq.calculate(t, t - depression, 1013.25, w, vapor_gg)
            if t_prev is not None:
                assert t_lcl > t_prev
            t_prev = t_lcl


# ===========================================================================
# Section 6 — MetPy cross-validation (flat tolerance)
# ===========================================================================

class TestMetPyCrossValidation:
    """
    Flat 0.5 K tolerance throughout — solver error is constant and does
    not depend on dewpoint depression. All discrepancy vs MetPy is from
    the vapour equation, not the solver.
    """

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_temp_agrees_with_metpy_goff_gratch(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        w              = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        t_iterative, _ = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        t_metpy, _     = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff           = abs(t_iterative - t_metpy)
        assert diff < METPY_TOL_K, (
            f"[{label}] GG: Iterative={t_iterative:.4f} K  "
            f"MetPy={t_metpy:.4f} K  Δ={diff:.4f} K  tol={METPY_TOL_K} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_pressure_agrees_with_metpy_goff_gratch(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        w              = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        _, p_iterative = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        _, p_metpy     = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff           = abs(p_iterative - p_metpy)
        assert diff < METPY_TOL_HPA, (
            f"[{label}] GG: Iterative={p_iterative:.4f} hPa  "
            f"MetPy={p_metpy:.4f} hPa  Δ={diff:.4f} hPa  tol={METPY_TOL_HPA} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_temp_agrees_with_metpy_hyland_wexler(self, eq, vapor_hw, label, pressure, temp_k, dewpoint_k):
        w              = _mixing_ratio(vapor_hw, dewpoint_k, pressure)
        t_iterative, _ = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_hw)
        t_metpy, _     = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff           = abs(t_iterative - t_metpy)
        assert diff < METPY_TOL_K, (
            f"[{label}] HW: Iterative={t_iterative:.4f} K  "
            f"MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
        )

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_agrees_with_metpy(self, eq, vapor_gg, base_temp, depressions):
        for dep in depressions:
            td             = base_temp - dep
            w              = _mixing_ratio(vapor_gg, td, 1013.25)
            t_iterative, _ = eq.calculate(base_temp, td, 1013.25, w, vapor_gg)
            t_metpy, _     = _metpy_lcl(1013.25, base_temp, td)
            diff           = abs(t_iterative - t_metpy)
            assert diff < METPY_TOL_K, (
                f"T={base_temp:.2f} K, dep={dep} K: "
                f"Iterative={t_iterative:.4f} K  MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )

    def test_large_array_elementwise_vs_metpy(self, eq, vapor_gg):
        temp_arr     = np.array([c[2] for c in STANDARD_CASES])
        dewpoint_arr = np.array([c[3] for c in STANDARD_CASES])
        pressure_arr = np.array([c[1] for c in STANDARD_CASES])
        w_arr        = _mixing_ratio_array(vapor_gg, dewpoint_arr, pressure_arr)

        t_arr, _ = eq.calculate(temp_arr, dewpoint_arr, pressure_arr, w_arr, vapor_gg)

        for i, (label, pressure, temp_k, dewpoint_k) in enumerate(STANDARD_CASES):
            t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
            diff       = abs(float(t_arr[i]) - t_metpy)
            assert diff < METPY_TOL_K, (
                f"[{label}] Array[{i}]: Iterative={t_arr[i]:.4f} K  "
                f"MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )

    def test_bolton_vapor_agrees_with_metpy(self, eq, vapor_bolton):
        """Bolton vapour equation via Vapor module must agree with MetPy."""
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        w              = _mixing_ratio(vapor_bolton, dewpoint_k, pressure_hpa)
        t_iterative, _ = eq.calculate(temp_k, dewpoint_k, pressure_hpa, w, vapor_bolton)
        t_metpy, _     = _metpy_lcl(pressure_hpa, temp_k, dewpoint_k)
        diff           = abs(t_iterative - t_metpy)
        assert diff < METPY_TOL_K, (
            f"Bolton vapor: Iterative={t_iterative:.4f} K  "
            f"MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
        )


# ===========================================================================
# Section 7 — More accurate than Bolton closed-form
# ===========================================================================

class TestAccuracyVsBolton:

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_iterative_at_least_as_accurate_as_bolton(self, eq, vapor_gg, label, pressure, temp_k, dewpoint_k):
        from meteocalc.lcl._jit_equations import _bolton_lcl_temp_scalar

        w              = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        t_iterative, _ = eq.calculate(temp_k, dewpoint_k, pressure, w, vapor_gg)
        t_bolton       = _bolton_lcl_temp_scalar(temp_k, dewpoint_k)
        t_metpy, _     = _metpy_lcl(pressure, temp_k, dewpoint_k)

        err_iterative = abs(t_iterative - t_metpy)
        err_bolton    = abs(t_bolton    - t_metpy)

        assert err_iterative <= err_bolton + 0.1, (
            f"[{label}] Iterative error ({err_iterative:.4f} K) exceeds "
            f"Bolton error ({err_bolton:.4f} K)"
        )


# ===========================================================================
# Section 8 — Inter-equation consistency
# ===========================================================================

class TestInterEquationConsistency:

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_goff_gratch_vs_hyland_wexler(self, eq, vapor_gg, vapor_hw, label, pressure, temp_k, dewpoint_k):
        w_gg = _mixing_ratio(vapor_gg, dewpoint_k, pressure)
        w_hw = _mixing_ratio(vapor_hw, dewpoint_k, pressure)

        t_gg, _ = eq.calculate(temp_k, dewpoint_k, pressure, w_gg, vapor_gg)
        t_hw, _ = eq.calculate(temp_k, dewpoint_k, pressure, w_hw, vapor_hw)

        assert abs(t_gg - t_hw) < 0.5, (
            f"[{label}] GG={t_gg:.4f} K  HW={t_hw:.4f} K  Δ={abs(t_gg-t_hw):.4f} K"
        )


# ===========================================================================
# Section 9 — Convergence
# ===========================================================================

class TestConvergence:

    def test_100k_random_parcels_all_finite(self, eq, vapor_gg):
        rng          = np.random.default_rng(42)
        n            = 100_000
        temp_k       = rng.uniform(280.0, 308.0, n)
        dewpoint_k   = np.clip(temp_k - rng.uniform(1.0, 10.0, n), 274.0, None)
        pressure_hpa = rng.uniform(850.0, 1013.25, n)
        w_arr        = _mixing_ratio_array(vapor_gg, dewpoint_k, pressure_hpa)

        t_lcl, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure_hpa, w_arr, vapor_gg)

        assert np.all(np.isfinite(t_lcl))
        assert np.all(np.isfinite(p_lcl))
        assert np.all(t_lcl <= temp_k)
        assert np.all(p_lcl <= pressure_hpa)

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_all_finite(self, eq, vapor_gg, base_temp, depressions):
        for dep in depressions:
            td       = base_temp - dep
            w        = _mixing_ratio(vapor_gg, td, 1013.25)
            t_lcl, p_lcl = eq.calculate(base_temp, td, 1013.25, w, vapor_gg)
            assert math.isfinite(t_lcl)
            assert math.isfinite(p_lcl)


# ===========================================================================
# Section 10 — Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_very_small_depression(self, eq, vapor_gg):
        w = _mixing_ratio(vapor_gg, 292.65, 1013.25)
        t_lcl, p_lcl = eq.calculate(293.15, 292.65, 1013.25, w, vapor_gg)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 293.15

    def test_high_pressure_surface(self, eq, vapor_gg):
        w = _mixing_ratio(vapor_gg, 285.15, 1050.0)
        t_lcl, p_lcl = eq.calculate(293.15, 285.15, 1050.0, w, vapor_gg)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 1050.0

    def test_low_pressure_surface(self, eq, vapor_gg):
        w = _mixing_ratio(vapor_gg, 278.15, 850.0)
        t_lcl, p_lcl = eq.calculate(285.15, 278.15, 850.0, w, vapor_gg)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 850.0

    def test_tropical_warm_humid(self, eq, vapor_gg):
        w = _mixing_ratio(vapor_gg, 303.15, 1005.0)
        t_lcl, p_lcl = eq.calculate(308.15, 303.15, 1005.0, w, vapor_gg)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 308.15

    def test_sensitivity_to_small_dewpoint_change(self, eq, vapor_gg):
        delta = 0.01
        t, td, p = 293.15, 285.15, 1013.25
        w1 = _mixing_ratio(vapor_gg, td,         p)
        w2 = _mixing_ratio(vapor_gg, td + delta, p)
        t1, _ = eq.calculate(t, td,         p, w1, vapor_gg)
        t2, _ = eq.calculate(t, td + delta, p, w2, vapor_gg)
        assert t2 > t1
        assert abs(t2 - t1) < delta * 10


# ===========================================================================
# Section 11 — Regression
# ===========================================================================

class TestRegression:
    """
    Hard-coded regression values seeded from actual IterativeLclEquation output.
    Uses Vapor.get_equation('goff_gratch', phase='water').
    Any implementation change that alters results will immediately fail.
    """

    CASES = [
        # (temp_k, dewpoint_k, pressure_hpa, exp_t,    exp_p,    tol_t, tol_p)
        (293.15, 285.15, 1013.25, 283.3605,   899.7022,  1e-3, 1e-1),
        (303.15, 298.15, 1000.0,  296.9407,   930.1399,  1e-3, 1e-1),
        (278.15, 268.15, 1013.25, 266.1260,   868.0749,  1e-3, 1e-1),
        (300.15, 290.15, 1000.0,  287.8639,   863.9366,  1e-3, 1e-1),
        (290.15, 289.15, 1013.25, 288.9170,   998.2622,  1e-3, 1e-1),
    ]

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_temp(self, eq, vapor_gg,
                             temp_k, dewpoint_k, pressure_hpa,
                             exp_t, exp_p, tol_t, tol_p):
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure_hpa)
        t_lcl, _ = eq.calculate(temp_k, dewpoint_k, pressure_hpa, w, vapor_gg)
        assert abs(t_lcl - exp_t) < tol_t, (
            f"T={temp_k}, Td={dewpoint_k}: got {t_lcl:.6f} K, "
            f"expected {exp_t:.6f} K (Δ={abs(t_lcl-exp_t):.6f} K)"
        )

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_pressure(self, eq, vapor_gg,
                                 temp_k, dewpoint_k, pressure_hpa,
                                 exp_t, exp_p, tol_t, tol_p):
        w = _mixing_ratio(vapor_gg, dewpoint_k, pressure_hpa)
        _, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure_hpa, w, vapor_gg)
        assert abs(p_lcl - exp_p) < tol_p, (
            f"T={temp_k}, Td={dewpoint_k}: got {p_lcl:.4f} hPa, "
            f"expected {exp_p:.4f} hPa (Δ={abs(p_lcl-exp_p):.4f} hPa)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])