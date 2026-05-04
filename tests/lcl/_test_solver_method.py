"""
Comprehensive unit tests for get_lcl_using_solver and _get_lcl_objective_function
==================================================================================
Cross-validated against MetPy's LCL solver (Romps 2017) to ensure accuracy.

The iterative solver finds the root of:

    f(T_LCL) = rs(T_LCL, p_LCL) - w = 0

where rs is the saturation mixing ratio at the LCL and w is the surface
mixing ratio. Accuracy is expected to within ~1e-8 K of the exact solution.

Tolerances
----------
- vs MetPy (Romps 2017 exact):  < 0.5 K typical, < 1.0 K for extreme cases
- vs Bolton closed-form:         < 1.5 K (Bolton's stated accuracy)
- Convergence:                   100% for physically valid inputs
- Inter-equation agreement:      < 0.5 K between Goff-Gratch and Hyland-Wexler

Dependencies
------------
    pip install pytest numpy numba metpy rapid-roots

References
----------
.. [1] Romps, D. M. (2017). Exact analytic solutions for pseudo-adiabatic
   ascent. Journal of the Atmospheric Sciences, 74(9), 3033-3039.
   https://doi.org/10.1175/JAS-D-17-0073.1

.. [2] Bolton, D. (1980). The computation of equivalent potential temperature.
   Monthly Weather Review, 108(7), 1046-1053.
"""

import math

import numpy as np
import pytest

from meteocalc.lcl._jit_equations import _bolton_lcl_temp_scalar
from meteocalc.lcl._solver_method import (
    _get_lcl_objective_function,
    get_lcl_using_solver,
)
from meteocalc.shared.constants import eps
from meteocalc.vapor._vapor_equations import GoffGratchEquation, HylandWexlerEquation

# ---------------------------------------------------------------------------
# ---- MetPy helper ---------------------------------------------------------
# ---------------------------------------------------------------------------

metpy = pytest.importorskip("metpy", reason="MetPy not installed — skipping MetPy tests")


def _metpy_lcl_temp(pressure_hpa: float, temp_k: float, dewpoint_k: float) -> float:
    """Return MetPy LCL temperature (K) using Romps (2017) exact solver."""
    from metpy.calc import lcl
    from metpy.units import units

    _, lcl_temp = lcl(
        pressure_hpa * units.hPa,
        temp_k * units.kelvin,
        dewpoint_k * units.kelvin,
    )
    return float(lcl_temp.to("kelvin").magnitude)


# ---------------------------------------------------------------------------
# ---- Shared fixtures ------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def goff_gratch_water():
    return GoffGratchEquation(surface_type="water")


@pytest.fixture(scope="module")
def hyland_wexler_water():
    return HylandWexlerEquation(surface_type="water")


@pytest.fixture(scope="module")
def goff_gratch_auto():
    return GoffGratchEquation(surface_type="automatic")


def _compute_mixing_ratio(vapor_eq, dewpoint_k, pressure_hpa):
    """Compute surface mixing ratio using the given vapour equation."""
    e = float(vapor_eq.calculate(dewpoint_k))
    return eps * e / (pressure_hpa - e)


def _compute_mixing_ratio_array(vapor_eq, dewpoint_arr, pressure_arr):
    """Compute surface mixing ratio array using the given vapour equation."""
    e_arr = vapor_eq.calculate(dewpoint_arr).astype(np.float64)
    return eps * e_arr / (pressure_arr - e_arr)


# ---------------------------------------------------------------------------
# ---- Test data ------------------------------------------------------------
# ---------------------------------------------------------------------------

# (label, pressure_hpa, temp_k, dewpoint_k)
STANDARD_CASES = [
    ("standard_midlat",    1013.25, 293.15, 285.15),  # 20°C / 12°C
    ("warm_humid",         1000.0,  303.15, 298.15),  # 30°C / 25°C — tropical
    ("cool_dry",           1013.25, 278.15, 270.15),  # 5°C / -3°C
    ("near_saturated",     1013.25, 290.15, 289.15),  # 1 K depression
    ("high_elevation",     850.0,   285.15, 278.15),  # 12°C / 5°C
    ("summer_continental", 1000.0,  300.15, 290.15),  # 27°C / 17°C
    ("autumn_uk",          1013.25, 283.15, 278.15),  # 10°C / 5°C
    ("moderate_dry",       1013.25, 295.15, 282.15),  # 22°C / 9°C
]

# Dewpoint depressions to sweep at fixed base temperatures
DEPRESSION_SWEEP = [
    (273.15 + 15, [1, 2, 5, 10, 15]),  # 15°C base
    (273.15 + 25, [1, 2, 5, 10, 15]),  # 25°C base
]


# ===========================================================================
# Section 1 — Physical constraints
# ===========================================================================

class TestPhysicalConstraints:
    """LCL temperature must satisfy fundamental physical laws."""

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_below_surface_temp(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """T_LCL must be ≤ T_surface — parcel cools on dry adiabatic ascent."""
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        assert converged
        assert t_lcl <= temp_k + 1e-9, (
            f"[{label}] T_LCL={t_lcl:.4f} K > T_surface={temp_k:.4f} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_above_absolute_zero(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """T_LCL must be physically positive."""
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        assert converged
        assert t_lcl > 0.0, f"[{label}] T_LCL={t_lcl:.4f} K ≤ 0 K"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_result_is_finite(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """Result must be finite — no NaN or inf."""
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        t_lcl, _, _ = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        assert math.isfinite(t_lcl), f"[{label}] T_LCL is not finite: {t_lcl}"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_objective_function_zero_at_root(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        assert converged

        obj            = _get_lcl_objective_function(goff_gratch_water)
        below_freezing = bool(temp_k < 273.15)   # ← matches solver logic
        residual       = obj(t_lcl, w, pressure, temp_k, below_freezing)

        assert abs(residual) < 1e-8, (
            f"[{label}] Objective not zero at root: f(T_LCL)={residual:.2e}"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_bracket_straddles_root(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """
        The bracket [a, b] = [Td - (T - Td), T] must straddle the root —
        f(a) and f(b) must have opposite signs for Brent to work.
        """
        w   = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        obj = _get_lcl_objective_function(goff_gratch_water)

        a = dewpoint_k - (temp_k - dewpoint_k)
        b = temp_k

        # below_freezing determined by surface temperature
        below_freezing = bool(temp_k < 273.15)

        f_a = obj(a, w, pressure, temp_k, below_freezing)
        f_b = obj(b, w, pressure, temp_k, below_freezing)

        assert f_a * f_b < 0, (
            f"[{label}] Bracket does not straddle root: "
            f"f(a)={f_a:.6f}, f(b)={f_b:.6f}, signs must differ"
        )


# ===========================================================================
# Section 2 — Convergence
# ===========================================================================

class TestConvergence:
    """Solver must converge for all physically valid inputs."""

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_scalar_converges(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """All standard cases must converge."""
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        _, iters, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        assert converged, f"[{label}] Did not converge after {iters} iterations"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_iterations_reasonable(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """
        Brent should converge in ≤ 20 iterations for well-bracketed problems.
        Excessive iterations indicate a bracket or objective function issue.
        """
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        _, iters, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        assert converged
        assert iters <= 20, (
            f"[{label}] Too many iterations: {iters} — bracket may be suboptimal"
        )

    def test_large_array_full_convergence(self, goff_gratch_auto):
        """100K random physically valid parcels must all converge."""
        rng          = np.random.default_rng(42)
        n            = 100_000
        temp_k       = rng.uniform(280.0, 310.0, n)
        dewpoint_k   = temp_k - rng.uniform(1.0, 10.0, n)
        pressure_hpa = rng.uniform(850.0, 1013.25, n)
        mixing_ratio = _compute_mixing_ratio_array(goff_gratch_auto, dewpoint_k, pressure_hpa)

        _, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, mixing_ratio, goff_gratch_auto
        )
        n_failed = (~converged).sum()
        assert n_failed == 0, f"{n_failed} / {n} parcels did not converge"

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_all_converge(self, base_temp, depressions, goff_gratch_water):
        """All depression values must converge at both base temperatures."""
        for dep in depressions:
            td = base_temp - dep
            w  = _compute_mixing_ratio(goff_gratch_water, td, 1013.25)
            _, iters, converged = get_lcl_using_solver(
                base_temp, td, 1013.25, w, goff_gratch_water
            )
            assert converged, (
                f"T={base_temp:.2f} K, dep={dep} K: did not converge after {iters} iterations"
            )


# ===========================================================================
# Section 3 — Scalar / array API
# ===========================================================================

class TestAPIBehaviour:
    """Scalar inputs return scalars; array inputs return arrays."""

    def test_scalar_input_returns_python_types(self, goff_gratch_water):
        """Scalar inputs must return Python float, int, bool — not numpy types."""
        w = _compute_mixing_ratio(goff_gratch_water, 285.15, 1013.25)
        t_lcl, iters, converged = get_lcl_using_solver(
            293.15, 285.15, 1013.25, w, goff_gratch_water
        )
        assert isinstance(t_lcl,    float), f"Expected float, got {type(t_lcl)}"
        assert isinstance(iters,    int),   f"Expected int, got {type(iters)}"
        assert isinstance(converged, bool),  f"Expected bool, got {type(converged)}"

    def test_array_input_returns_numpy_arrays(self, goff_gratch_water):
        """Array inputs must return numpy arrays of correct dtype and shape."""
        n            = 5
        temp_arr     = np.full(n, 293.15)
        dewpoint_arr = np.full(n, 285.15)
        pressure_arr = np.full(n, 1013.25)
        w_arr        = _compute_mixing_ratio_array(goff_gratch_water, dewpoint_arr, pressure_arr)

        t_lcl, iters, converged = get_lcl_using_solver(
            temp_arr, dewpoint_arr, pressure_arr, w_arr, goff_gratch_water
        )

        assert isinstance(t_lcl,    np.ndarray)
        assert isinstance(iters,    np.ndarray)
        assert isinstance(converged, np.ndarray)
        assert t_lcl.shape    == (n,)
        assert iters.shape    == (n,)
        assert converged.shape == (n,)
        assert t_lcl.dtype    == np.float64

    def test_single_element_array_returns_array(self, goff_gratch_water):
        """1-element array input must return arrays, not scalars."""
        temp_arr     = np.array([293.15])
        dewpoint_arr = np.array([285.15])
        pressure_arr = np.array([1013.25])
        w_arr        = _compute_mixing_ratio_array(goff_gratch_water, dewpoint_arr, pressure_arr)

        t_lcl, iters, converged = get_lcl_using_solver(
            temp_arr, dewpoint_arr, pressure_arr, w_arr, goff_gratch_water
        )

        assert isinstance(t_lcl, np.ndarray)
        assert t_lcl.shape == (1,)

    def test_scalar_array_results_consistent(self, goff_gratch_water):
        """Scalar and array calls with same inputs must give identical results."""
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)

        t_scalar, _, _ = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        t_array, _, _ = get_lcl_using_solver(
            np.array([temp_k]),
            np.array([dewpoint_k]),
            np.array([pressure_hpa]),
            np.array([w]),
            goff_gratch_water,
        )

        assert abs(t_scalar - float(t_array[0])) < 1e-10, (
            f"Scalar={t_scalar:.8f} K vs array={float(t_array[0]):.8f} K — should be identical"
        )

    def test_deterministic_repeated_calls(self, goff_gratch_water):
        """JIT-compiled solver must return identical results on repeated calls."""
        w = _compute_mixing_ratio(goff_gratch_water, 285.15, 1013.25)
        results = [
            get_lcl_using_solver(293.15, 285.15, 1013.25, w, goff_gratch_water)[0]
            for _ in range(5)
        ]
        assert all(r == results[0] for r in results), "Non-deterministic output detected"


# ===========================================================================
# Section 4 — Monotonicity
# ===========================================================================

class TestMonotonicity:
    """LCL temperature must respond correctly to changes in inputs."""

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl(self, base_temp, depressions, goff_gratch_water):
        """
        Increasing dewpoint depression must lower T_LCL — a drier parcel
        must ascend further before reaching saturation.
        """
        t_lcl_prev = None
        for dep in sorted(depressions):
            td = base_temp - dep
            w  = _compute_mixing_ratio(goff_gratch_water, td, 1013.25)
            t_lcl, _, converged = get_lcl_using_solver(
                base_temp, td, 1013.25, w, goff_gratch_water
            )
            assert converged
            if t_lcl_prev is not None:
                assert t_lcl < t_lcl_prev, (
                    f"T={base_temp:.2f} K, dep={dep} K: T_LCL did not decrease "
                    f"(got {t_lcl:.4f} K, prev={t_lcl_prev:.4f} K)"
                )
            t_lcl_prev = t_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_warmer_parcel_gives_warmer_lcl(self, depression, goff_gratch_water):
        """At fixed depression, a warmer parcel must give a warmer T_LCL."""
        temps      = [278.15, 283.15, 288.15, 293.15, 298.15]
        t_lcl_prev = None
        for t in temps:
            td = t - depression
            w  = _compute_mixing_ratio(goff_gratch_water, td, 1013.25)
            t_lcl, _, converged = get_lcl_using_solver(
                t, td, 1013.25, w, goff_gratch_water
            )
            assert converged
            if t_lcl_prev is not None:
                assert t_lcl > t_lcl_prev, (
                    f"depression={depression} K: warmer parcel did not give warmer T_LCL "
                    f"(got {t_lcl:.4f} K, prev={t_lcl_prev:.4f} K)"
                )
            t_lcl_prev = t_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_lcl_pressure_independent_of_surface_pressure(self, depression, goff_gratch_water):
        """
        LCL temperature is primarily determined by T and Td, not surface
        pressure. Verify result is physically reasonable across pressure range.
        """
        temp_k     = 293.15
        dewpoint_k = temp_k - depression
        pressures  = [850.0, 900.0, 950.0, 1013.25]

        results = []
        for p in pressures:
            w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, p)
            t_lcl, _, converged = get_lcl_using_solver(
                temp_k, dewpoint_k, p, w, goff_gratch_water
            )
            assert converged
            assert math.isfinite(t_lcl)
            assert t_lcl <= temp_k
            results.append(t_lcl)

        # All results should be within 1 K of each other — pressure effect is small
        assert max(results) - min(results) < 1.0, (
            f"T_LCL varies by {max(results)-min(results):.4f} K across pressure range — "
            f"unexpectedly large pressure sensitivity"
        )


# ===========================================================================
# Section 5 — Cross-validation against MetPy (Romps 2017)
# ===========================================================================

class TestMetPyCrossValidation:
    """
    Validate iterative solver against MetPy's Romps (2017) exact solution.

    MetPy uses an exact analytical LCL solver (Romps 2017), so all
    discrepancy here is from the vapour pressure equation choice, not
    the root-finding method. Expected agreement is < 0.5 K for typical
    conditions.

    Tolerances
    ----------
    Near-saturated (ΔT < 3 K):    < 0.1 K
    Typical (ΔT 3–10 K):          < 0.5 K
    Moderate dry (ΔT 10–15 K):    < 0.5 K
    """

    TOL_NEAR_SATURATED = 0.1   # K
    TOL_TYPICAL        = 0.5   # K

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_agrees_with_metpy(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """Iterative solver must agree with MetPy within tiered tolerances."""
        depression = temp_k - dewpoint_k
        tol = self.TOL_NEAR_SATURATED if depression < 3.0 else self.TOL_TYPICAL

        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        reference = _metpy_lcl_temp(pressure, temp_k, dewpoint_k)
        diff = abs(t_lcl - reference)

        assert converged
        assert diff < tol, (
            f"[{label}] Iterative={t_lcl:.4f} K  MetPy={reference:.4f} K  "
            f"Δ={diff:.4f} K  depression={depression:.1f} K  tol={tol} K"
        )

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_agrees_with_metpy(self, base_temp, depressions, goff_gratch_water):
        """Sweep dewpoint depressions and compare against MetPy at each step."""
        for dep in depressions:
            td  = base_temp - dep
            tol = self.TOL_NEAR_SATURATED if dep < 3.0 else self.TOL_TYPICAL
            w   = _compute_mixing_ratio(goff_gratch_water, td, 1013.25)

            t_lcl, _, converged = get_lcl_using_solver(
                base_temp, td, 1013.25, w, goff_gratch_water
            )
            reference = _metpy_lcl_temp(1013.25, base_temp, td)
            diff      = abs(t_lcl - reference)

            assert converged
            assert diff < tol, (
                f"T={base_temp:.2f} K, dep={dep} K: "
                f"Iterative={t_lcl:.4f} K  MetPy={reference:.4f} K  Δ={diff:.4f} K"
            )

    def test_tropical_near_saturated_close_to_metpy(self, goff_gratch_water):
        """Tropical near-saturated conditions — tightest agreement expected."""
        cases = [
            (1010.0, 303.15, 301.15),   # 30°C / 28°C
            (1010.0, 308.15, 306.15),   # 35°C / 33°C
        ]
        for p, t, td in cases:
            w         = _compute_mixing_ratio(goff_gratch_water, td, p)
            t_lcl, _, converged = get_lcl_using_solver(t, td, p, w, goff_gratch_water)
            reference = _metpy_lcl_temp(p, t, td)
            assert converged
            assert abs(t_lcl - reference) < self.TOL_NEAR_SATURATED, (
                f"Tropical: Iterative={t_lcl:.4f} K  MetPy={reference:.4f} K"
            )

    def test_hyland_wexler_agrees_with_metpy(self, hyland_wexler_water):
        """Hyland-Wexler equation must also agree with MetPy."""
        cases = STANDARD_CASES[:4]
        for label, pressure, temp_k, dewpoint_k in cases:
            depression = temp_k - dewpoint_k
            tol = self.TOL_NEAR_SATURATED if depression < 3.0 else self.TOL_TYPICAL

            w = _compute_mixing_ratio(hyland_wexler_water, dewpoint_k, pressure)
            t_lcl, _, converged = get_lcl_using_solver(
                temp_k, dewpoint_k, pressure, w, hyland_wexler_water
            )
            reference = _metpy_lcl_temp(pressure, temp_k, dewpoint_k)
            diff = abs(t_lcl - reference)

            assert converged
            assert diff < tol, (
                f"[{label}] HW: Iterative={t_lcl:.4f} K  MetPy={reference:.4f} K  "
                f"Δ={diff:.4f} K"
            )


# ===========================================================================
# Section 6 — Inter-equation consistency
# ===========================================================================

class TestInterEquationConsistency:
    """
    Goff-Gratch and Hyland-Wexler should agree closely for the same inputs
    since both are high-accuracy vapour pressure equations.
    """

    INTER_EQUATION_TOL = 0.5   # K

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_goff_gratch_vs_hyland_wexler(
        self, label, pressure, temp_k, dewpoint_k,
        goff_gratch_water, hyland_wexler_water
    ):
        """Goff-Gratch and Hyland-Wexler must agree within 0.5 K."""
        w_gg = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        w_hw = _compute_mixing_ratio(hyland_wexler_water, dewpoint_k, pressure)

        t_gg, _, c_gg = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w_gg, goff_gratch_water
        )
        t_hw, _, c_hw = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w_hw, hyland_wexler_water
        )

        assert c_gg and c_hw
        diff = abs(t_gg - t_hw)
        assert diff < self.INTER_EQUATION_TOL, (
            f"[{label}] GG={t_gg:.4f} K  HW={t_hw:.4f} K  Δ={diff:.4f} K"
        )


# ===========================================================================
# Section 7 — Bolton cross-check
# ===========================================================================

class TestBoltonCrossCheck:
    """
    Iterative solver must agree with Bolton (1980) closed-form within
    Bolton's stated accuracy (~1 K for typical conditions).
    """

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_agrees_with_bolton(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """Iterative solver must agree with Bolton eq. 15 within 1.5 K."""
        depression = temp_k - dewpoint_k
        tol = 0.5 if depression < 5.0 else 1.5

        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        t_iterative, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, goff_gratch_water
        )
        t_bolton = _bolton_lcl_temp_scalar(temp_k, dewpoint_k)
        diff     = abs(t_iterative - t_bolton)

        assert converged
        assert diff < tol, (
            f"[{label}] Iterative={t_iterative:.4f} K  Bolton={t_bolton:.4f} K  "
            f"Δ={diff:.4f} K  tol={tol} K"
        )

    def test_iterative_more_accurate_than_bolton(self, goff_gratch_water):
        """
        Iterative solver should be closer to MetPy (exact) than Bolton
        for at least the majority of standard cases — validates that the
        iterative method adds value over the closed-form approximation.
        """
        iterative_wins = 0
        for label, pressure, temp_k, dewpoint_k in STANDARD_CASES:
            w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
            t_iterative, _, converged = get_lcl_using_solver(
                temp_k, dewpoint_k, pressure, w, goff_gratch_water
            )
            t_bolton  = _bolton_lcl_temp_scalar(temp_k, dewpoint_k)
            reference = _metpy_lcl_temp(pressure, temp_k, dewpoint_k)

            if converged:
                err_iterative = abs(t_iterative - reference)
                err_bolton    = abs(t_bolton    - reference)
                if err_iterative < err_bolton:
                    iterative_wins += 1

        assert iterative_wins >= len(STANDARD_CASES) // 2, (
            f"Iterative solver only outperformed Bolton on "
            f"{iterative_wins}/{len(STANDARD_CASES)} cases"
        )


# ===========================================================================
# Section 8 — Objective function tests
# ===========================================================================

class TestObjectiveFunction:
    """Tests for _get_lcl_objective_function directly."""

    def test_returns_callable(self, goff_gratch_water):
        """Must return a callable."""
        obj = _get_lcl_objective_function(goff_gratch_water)
        assert callable(obj)

    def test_objective_zero_at_known_root(self, goff_gratch_water):
        """
        Objective function must evaluate to ≈ 0 at the known LCL temperature
        — verifies the mathematical formulation is correct.
        """
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        w   = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        obj = _get_lcl_objective_function(goff_gratch_water)

        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        assert converged

        below_freezing = bool(t_lcl < 273.15)
        residual       = obj(t_lcl, w, pressure_hpa, temp_k, below_freezing)
        assert abs(residual) < 1e-8, (
            f"f(T_LCL) = {residual:.2e} — should be ≈ 0 at the root"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_bracket_signs_correct(self, label, pressure, temp_k, dewpoint_k, goff_gratch_water):
        """
        f(a) must be negative and f(b) positive for Brent to converge.
        Validates the physical bracket construction.
        """
        w              = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure)
        obj            = _get_lcl_objective_function(goff_gratch_water)
        below_freezing = bool(temp_k < 273.15)

        a = dewpoint_k - (temp_k - dewpoint_k)
        b = temp_k

        f_a = obj(a, w, pressure, temp_k, below_freezing)
        f_b = obj(b, w, pressure, temp_k, below_freezing)

        assert f_a < 0, (
            f"[{label}] f(a) = {f_a:.6f} — must be negative for valid bracket"
        )
        assert f_b > 0, (
            f"[{label}] f(b) = {f_b:.6f} — must be positive for valid bracket"
        )

    def test_different_equations_give_different_objectives(
        self, goff_gratch_water, hyland_wexler_water
    ):
        """
        Different vapour equations must produce different objective functions
        that evaluate to different values at the same non-root point.
        """
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        w_gg           = _compute_mixing_ratio(goff_gratch_water,    dewpoint_k, pressure_hpa)
        w_hw           = _compute_mixing_ratio(hyland_wexler_water,  dewpoint_k, pressure_hpa)
        below_freezing = bool(temp_k < 273.15)

        obj_gg = _get_lcl_objective_function(goff_gratch_water)
        obj_hw = _get_lcl_objective_function(hyland_wexler_water)

        # Evaluate at a non-root point — midpoint of bracket
        x    = (dewpoint_k + temp_k) / 2
        f_gg = obj_gg(x, w_gg, pressure_hpa, temp_k, below_freezing)
        f_hw = obj_hw(x, w_hw, pressure_hpa, temp_k, below_freezing)

        assert f_gg != f_hw, (
            "Different vapour equations produced identical objective values — "
            "constants may not be captured correctly in closure"
        )

    def test_below_freezing_uses_ice_constants(self, goff_gratch_water):
        """
        Below 273.15 K, ice constants must be used — verified by comparing
        objective values with below_freezing=True vs False at a sub-freezing
        candidate temperature.
        """
        temp_k, dewpoint_k, pressure_hpa = 263.15, 258.15, 1013.25
        w              = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        obj            = _get_lcl_objective_function(goff_gratch_water)

        x = 260.0   # sub-freezing candidate

        f_ice   = obj(x, w, pressure_hpa, temp_k, True)    # ice constants
        f_water = obj(x, w, pressure_hpa, temp_k, False)   # water constants

        # Ice and water es differ at sub-freezing temps — objective values must differ
        assert f_ice != f_water, (
            f"Ice and water constants produced identical objective at {x} K — "
            f"below_freezing branch may not be working"
        )

    def test_above_freezing_uses_water_constants(self, goff_gratch_water):
        """
        Above 273.15 K, water constants must be used — verified by confirming
        below_freezing=False gives a physically consistent result vs MetPy.
        """
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        w   = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        obj = _get_lcl_objective_function(goff_gratch_water)

        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        assert converged
        assert t_lcl > 273.15, "Expected above-freezing LCL for this case"

        # With correct water constants, residual must be ≈ 0 at root
        residual = obj(t_lcl, w, pressure_hpa, temp_k, False)
        assert abs(residual) < 1e-8, (
            f"Water constants gave residual {residual:.2e} at root — "
            f"expected ≈ 0"
        )


# ===========================================================================
# Section 9 — Edge cases
# ===========================================================================

class TestEdgeCases:
    """Boundary and edge case inputs."""

    def test_very_small_depression(self, goff_gratch_water):
        """0.5 K depression — nearly saturated parcel must converge."""
        temp_k, dewpoint_k, pressure_hpa = 293.15, 292.65, 1013.25
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        t_lcl, iters, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        assert converged, f"0.5 K depression did not converge after {iters} iterations"
        assert math.isfinite(t_lcl)
        assert t_lcl <= temp_k

    def test_high_pressure_surface(self, goff_gratch_water):
        """1050 hPa surface pressure — valid but at upper bound."""
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1050.0
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        assert converged
        assert math.isfinite(t_lcl)
        assert t_lcl <= temp_k

    def test_low_pressure_surface(self, goff_gratch_water):
        """850 hPa surface — elevated terrain."""
        temp_k, dewpoint_k, pressure_hpa = 285.15, 278.15, 850.0
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        assert converged
        assert math.isfinite(t_lcl)
        assert t_lcl <= temp_k

    def test_warm_tropical_surface(self, goff_gratch_water):
        """35°C / 32°C — hot humid tropical surface."""
        temp_k, dewpoint_k, pressure_hpa = 308.15, 305.15, 1005.0
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        assert converged
        assert math.isfinite(t_lcl)

    def test_sensitivity_to_small_dewpoint_change(self, goff_gratch_water):
        """
        Small perturbation in Td must produce small, proportional change in T_LCL.
        Guards against numerical instability near the root.
        """
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        delta = 0.01  # 0.01 K perturbation

        w1 = _compute_mixing_ratio(goff_gratch_water, dewpoint_k,         pressure_hpa)
        w2 = _compute_mixing_ratio(goff_gratch_water, dewpoint_k + delta,  pressure_hpa)

        t1, _, c1 = get_lcl_using_solver(temp_k, dewpoint_k,        pressure_hpa, w1, goff_gratch_water)
        t2, _, c2 = get_lcl_using_solver(temp_k, dewpoint_k + delta, pressure_hpa, w2, goff_gratch_water)

        assert c1 and c2
        assert t2 > t1, "Increasing Td should increase T_LCL"
        assert abs(t2 - t1) < delta * 10, (
            f"T_LCL changed by {abs(t2-t1):.4f} K for {delta} K Td perturbation — "
            f"possible instability"
        )


# ===========================================================================
# Section 10 — Regression
# ===========================================================================

class TestRegression:
    """
    Hard-coded regression values seeded from actual solver output.
    Any implementation change that alters results will immediately fail here.

    Values generated with GoffGratchEquation(surface_type="water").
    """

    CASES = [
        # (temp_k,  dewpoint_k, pressure_hpa, expected_t_lcl, tol)
        (293.15,    285.15,     1013.25,       283.3605,       1e-3),
        (303.15,    298.15,     1000.0,        296.9407,       1e-3),
        (283.15,    278.15,     1013.25,       277.0650,       1e-3),
        (300.15,    290.15,     1000.0,        287.8639,       1e-3),
        (290.15,    289.15,     1013.25,       288.9170,       1e-3),  # near-saturated — looser
    ]

    @pytest.mark.parametrize("temp_k,dewpoint_k,pressure_hpa,expected,tol", CASES)
    def test_regression(self, temp_k, dewpoint_k, pressure_hpa, expected, tol, goff_gratch_water):
        w = _compute_mixing_ratio(goff_gratch_water, dewpoint_k, pressure_hpa)
        t_lcl, _, converged = get_lcl_using_solver(
            temp_k, dewpoint_k, pressure_hpa, w, goff_gratch_water
        )
        assert converged
        assert abs(t_lcl - expected) < tol, (
            f"T={temp_k}, Td={dewpoint_k}: got {t_lcl:.6f} K, "
            f"expected {expected:.6f} K (Δ={abs(t_lcl-expected):.6f} K)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])