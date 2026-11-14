"""
Tests for Lcl.get_lcl_using_approximation — public API
=======================================================

Cross-validates against MetPy's LCL implementation (Romps 2017) and verifies
the full public-API stack: enum/string parsing → approximation equation
dispatch → Bolton (1980) JIT evaluation → (T_LCL, P_LCL) output.

This file tests the *public interface* ``Lcl.get_lcl_using_approximation``.
The lower-level ``BoltonLclEquation.calculate`` is tested separately in
``_test_lcl_bolton_equation_class.py``.

Test sections
-------------
1.  Enum / string API — lcl_equation_name string and enum inputs
2.  Physical constraints — T_LCL ≤ T_surface, P_LCL ≤ P_surface, Poisson
3.  Return types — scalar → float, array → ndarray of float64
4.  MetPy cross-validation — tiered tolerances from Bolton (1980) Table 1
5.  Monotonicity — depression sweep, surface temperature sweep,
                   surface pressure sweep
6.  Convergence — 100 K random parcels all finite and physical
7.  Regression guard — hard-coded seeded values
8.  Consistency — ``get_lcl_using_approximation`` equals
                  ``BoltonLclEquation.calculate`` directly
9.  Edge cases — near-saturated, high/low pressure, tropical

Tolerance vs MetPy (Romps 2017 exact)
--------------------------------------
Bolton (1980) accuracy degrades with dewpoint depression. Tiered tolerances:

    depression < 3 K   →  0.3 K
    depression 3–10 K  →  0.8 K
    depression 10–20 K →  1.5 K
    depression > 20 K  →  2.5 K

Pressure tolerance is derived as: T_tol × 4.0 hPa/K (approximate d p/d T
along the Poisson relation at typical conditions).

References
----------
.. [1] Bolton, D. (1980). The computation of equivalent potential temperature.
   Monthly Weather Review, 108(7), 1046–1053.
.. [2] Romps, D. M. (2017). Exact analytic solutions for pseudo-adiabatic
   ascent. Journal of the Atmospheric Sciences, 74(9), 3033–3039.
   https://doi.org/10.1175/JAS-D-17-0073.1

Dependencies
------------
    pip install pytest numpy numba metpy
"""

import math

import numpy as np
import pytest

from meteocalc.lcl._enums import LclEquationName
from meteocalc.lcl._lcl_equation import BoltonLclEquation
from meteocalc.lcl.core import Lcl
from meteocalc.shared.constants import Rd, cpd

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

def _approx(temp_k, dewpoint_k, pressure_hpa, **kwargs):
    """Thin wrapper around ``Lcl.get_lcl_using_approximation``."""
    return Lcl.get_lcl_using_approximation(
        temp_k=temp_k,
        dewpoint_temp_k=dewpoint_k,
        pressure_hpa=pressure_hpa,
        **kwargs,
    )


def _metpy_tol(depression: float) -> float:
    """Tiered tolerance (K) based on Bolton (1980) stated accuracy."""
    if depression < 3.0:
        return 0.3
    elif depression < 10.0:
        return 0.8
    elif depression < 20.0:
        return 1.5
    return 2.5


# ---------------------------------------------------------------------------
# ---- Test data --------------------------------------------------------------
# ---------------------------------------------------------------------------

# (label, pressure_hpa, temp_k, dewpoint_k)
STANDARD_CASES = [
    ("standard_midlat",    1013.25, 293.15, 285.15),  # 20 °C / 12 °C — 8 K
    ("warm_humid",         1000.0,  303.15, 298.15),  # 30 °C / 25 °C — 5 K
    ("cool_dry",           1013.25, 278.15, 268.15),  # 5 °C / −5 °C  — 10 K
    ("near_saturated",     1013.25, 290.15, 289.15),  # 1 K depression
    ("high_elevation",     850.0,   285.15, 278.15),  # 12 °C / 5 °C
    ("summer_continental", 1000.0,  300.15, 290.15),  # 27 °C / 17 °C — 10 K
    ("autumn_uk",          1013.25, 283.15, 278.15),  # 10 °C / 5 °C  — 5 K
    ("moderate_dry",       1013.25, 295.15, 282.15),  # 22 °C / 9 °C  — 13 K
]

DEPRESSION_SWEEP = [
    (273.15 + 15, [1, 2, 5, 10, 15, 20]),
    (273.15 + 25, [1, 2, 5, 10, 15, 20]),
]


# ===========================================================================
# Section 1 — Enum / string API
# ===========================================================================

class TestEnumStringAPI:
    """
    ``lcl_equation_name`` must accept both a plain string and its
    :class:`~meteocalc.lcl._enums.LclEquationName` enum equivalent.
    Results must be bit-identical regardless of which form is used.
    """

    def test_string_bolton_equals_enum_bolton(self):
        t_s, p_s = _approx(293.15, 285.15, 1013.25, lcl_equation_name="bolton")
        t_e, p_e = _approx(293.15, 285.15, 1013.25, lcl_equation_name=LclEquationName.BOLTON)
        assert t_s == t_e
        assert p_s == p_e

    def test_default_equation_is_bolton(self):
        """Omitting ``lcl_equation_name`` must give the same result as ``'bolton'``."""
        t_default, p_default = _approx(293.15, 285.15, 1013.25)
        t_explicit, p_explicit = _approx(293.15, 285.15, 1013.25, lcl_equation_name="bolton")
        assert t_default == t_explicit
        assert p_default == p_explicit

    def test_string_and_enum_agree_array_input(self):
        """String/enum equivalence must hold for array inputs."""
        n    = 4
        t_a  = np.full(n, 293.15)
        td_a = np.full(n, 285.15)
        p_a  = np.full(n, 1013.25)
        t_s, p_s = Lcl.get_lcl_using_approximation(t_a, td_a, p_a, lcl_equation_name="bolton")
        t_e, p_e = Lcl.get_lcl_using_approximation(t_a, td_a, p_a, lcl_equation_name=LclEquationName.BOLTON)
        np.testing.assert_array_equal(t_s, t_e)
        np.testing.assert_array_equal(p_s, p_e)

    def test_invalid_equation_name_raises(self):
        """An unrecognised equation name must raise ``ValueError`` or ``KeyError``."""
        with pytest.raises((ValueError, KeyError)):
            _approx(293.15, 285.15, 1013.25, lcl_equation_name="nonexistent")

    def test_invalid_equation_name_not_in_registry_raises(self):
        """A valid enum that is not in the approximation registry must raise."""
        with pytest.raises((ValueError, KeyError)):
            _approx(293.15, 285.15, 1013.25, lcl_equation_name="iterative")


# ===========================================================================
# Section 2 — Physical constraints
# ===========================================================================

class TestPhysicalConstraints:
    """LCL results must satisfy fundamental atmospheric physics."""

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_below_surface_temp(self, label, pressure, temp_k, dewpoint_k):
        """T_LCL ≤ T_surface — parcel cools on dry adiabatic ascent."""
        t_lcl, _ = _approx(temp_k, dewpoint_k, pressure)
        assert t_lcl <= temp_k + 1e-9, (
            f"[{label}] T_LCL={t_lcl:.4f} K > T_surface={temp_k:.4f} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_pressure_below_surface_pressure(self, label, pressure, temp_k, dewpoint_k):
        """P_LCL ≤ P_surface — LCL is above the surface."""
        _, p_lcl = _approx(temp_k, dewpoint_k, pressure)
        assert p_lcl <= pressure + 1e-9, (
            f"[{label}] P_LCL={p_lcl:.4f} hPa > P_surface={pressure:.4f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_above_absolute_zero(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, _ = _approx(temp_k, dewpoint_k, pressure)
        assert t_lcl > 0.0, f"[{label}] T_LCL={t_lcl:.4f} K ≤ 0"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_pressure_positive(self, label, pressure, temp_k, dewpoint_k):
        _, p_lcl = _approx(temp_k, dewpoint_k, pressure)
        assert p_lcl > 0.0, f"[{label}] P_LCL={p_lcl:.4f} hPa ≤ 0"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_results_are_finite(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, p_lcl = _approx(temp_k, dewpoint_k, pressure)
        assert math.isfinite(t_lcl), f"[{label}] T_LCL not finite"
        assert math.isfinite(p_lcl), f"[{label}] P_LCL not finite"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_poisson_relation_consistent(self, label, pressure, temp_k, dewpoint_k):
        """P_LCL must satisfy the Poisson relation given T_LCL."""
        t_lcl, p_lcl = _approx(temp_k, dewpoint_k, pressure)
        expected_p   = pressure * (t_lcl / temp_k) ** (cpd / Rd)
        assert abs(p_lcl - expected_p) < 1e-6, (
            f"[{label}] Poisson violated: got {p_lcl:.6f} hPa, "
            f"expected {expected_p:.6f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_below_surface_dewpoint(self, label, pressure, temp_k, dewpoint_k):
        """T_LCL < T_d: parcel cools at DALR (~9.8 K/km), dewpoint falls
        only ~1.8 K/km; they converge with T_LCL below the surface dewpoint."""
        t_lcl, _ = _approx(temp_k, dewpoint_k, pressure)
        assert t_lcl < dewpoint_k + 1e-9, (
            f"[{label}] T_LCL={t_lcl:.4f} K unexpectedly above T_d={dewpoint_k:.4f} K"
        )


# ===========================================================================
# Section 3 — Return types
# ===========================================================================

class TestReturnTypes:
    """Scalar inputs return Python floats; array inputs return float64 ndarrays."""

    def test_scalar_returns_floats(self):
        t, p = _approx(293.15, 285.15, 1013.25)
        assert isinstance(t, float), f"Expected float, got {type(t)}"
        assert isinstance(p, float), f"Expected float, got {type(p)}"

    def test_returns_two_values(self):
        assert len(_approx(293.15, 285.15, 1013.25)) == 2

    def test_array_returns_ndarrays(self):
        n    = 5
        t, p = Lcl.get_lcl_using_approximation(
            np.full(n, 293.15), np.full(n, 285.15), np.full(n, 1013.25)
        )
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)

    def test_array_dtype_float64(self):
        n    = 4
        t, p = Lcl.get_lcl_using_approximation(
            np.full(n, 293.15), np.full(n, 285.15), np.full(n, 1013.25)
        )
        assert t.dtype == np.float64
        assert p.dtype == np.float64

    def test_array_shape_matches_input(self):
        n    = 7
        t, p = Lcl.get_lcl_using_approximation(
            np.full(n, 293.15), np.full(n, 285.15), np.full(n, 1013.25)
        )
        assert t.shape == (n,)
        assert p.shape == (n,)

    def test_2d_array_shape_preserved(self):
        shape = (3, 4)
        t, p  = Lcl.get_lcl_using_approximation(
            np.full(shape, 293.15), np.full(shape, 285.15), np.full(shape, 1013.25)
        )
        assert t.shape == shape
        assert p.shape == shape

    def test_scalar_array_consistency(self):
        """Single-element array result must equal scalar result."""
        t_sc, p_sc = _approx(293.15, 285.15, 1013.25)
        t_ar, p_ar = Lcl.get_lcl_using_approximation(
            np.array([293.15]), np.array([285.15]), np.array([1013.25])
        )
        assert abs(t_sc - float(t_ar[0])) < 1e-10
        assert abs(p_sc - float(p_ar[0])) < 1e-10

    def test_mixed_scalar_array_broadcast(self):
        """Scalar + array inputs must broadcast to the array shape."""
        n    = 4
        t, p = Lcl.get_lcl_using_approximation(
            np.full(n, 293.15), 285.15, 1013.25
        )
        assert t.shape == (n,)
        assert p.shape == (n,)

    def test_deterministic(self):
        """Repeated calls with identical inputs must return identical results."""
        results = [_approx(293.15, 285.15, 1013.25) for _ in range(5)]
        for t, p in results:
            assert t == results[0][0]
            assert p == results[0][1]


# ===========================================================================
# Section 4 — MetPy cross-validation
# ===========================================================================

class TestMetPyCrossValidation:
    """
    Validate Bolton (1980) approximation against MetPy's Romps (2017) exact solver.
    All discrepancy is Bolton's analytical error — tolerances are tiered by
    dewpoint depression following Bolton (1980) Table 1.
    """

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_temp_agrees_with_metpy(self, label, pressure, temp_k, dewpoint_k):
        depression = temp_k - dewpoint_k
        tol        = _metpy_tol(depression)

        t_lcl, _   = _approx(temp_k, dewpoint_k, pressure)
        t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff       = abs(t_lcl - t_metpy)

        assert diff < tol, (
            f"[{label}] Bolton={t_lcl:.4f} K  MetPy={t_metpy:.4f} K  "
            f"Δ={diff:.4f} K  dep={depression:.1f} K  tol={tol} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_pressure_agrees_with_metpy(self, label, pressure, temp_k, dewpoint_k):
        depression = temp_k - dewpoint_k
        tol_hpa    = _metpy_tol(depression) * 4.0

        _, p_lcl   = _approx(temp_k, dewpoint_k, pressure)
        _, p_metpy = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff       = abs(p_lcl - p_metpy)

        assert diff < tol_hpa, (
            f"[{label}] Bolton={p_lcl:.4f} hPa  MetPy={p_metpy:.4f} hPa  "
            f"Δ={diff:.4f} hPa  tol={tol_hpa:.2f} hPa"
        )

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_agrees_with_metpy(self, base_temp, depressions):
        for dep in depressions:
            td         = base_temp - dep
            tol        = _metpy_tol(dep)
            t_lcl, _   = _approx(base_temp, td, 1013.25)
            t_metpy, _ = _metpy_lcl(1013.25, base_temp, td)
            diff       = abs(t_lcl - t_metpy)
            assert diff < tol, (
                f"T={base_temp:.2f} K, dep={dep} K: "
                f"Bolton={t_lcl:.4f} K  MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )

    def test_near_saturated_tight_agreement(self):
        """Near-saturated parcels (ΔT < 2 K) — Bolton is most accurate here."""
        cases = [
            (1013.25, 293.15, 292.15),
            (1013.25, 293.15, 291.65),
            (1013.25, 303.15, 302.15),
        ]
        for p, t, td in cases:
            t_lcl, _   = _approx(t, td, p)
            t_metpy, _ = _metpy_lcl(p, t, td)
            diff       = abs(t_lcl - t_metpy)
            assert diff < 0.3, (
                f"Near-saturated: Bolton={t_lcl:.4f} K  MetPy={t_metpy:.4f} K  "
                f"Δ={diff:.4f} K (tol=0.3 K)"
            )

    def test_tropical_conditions_agree(self):
        """Tropical warm/humid conditions."""
        cases = [
            (1010.0, 303.15, 298.15),
            (1010.0, 308.15, 303.15),
        ]
        for p, t, td in cases:
            dep        = t - td
            tol        = _metpy_tol(dep)
            t_lcl, _   = _approx(t, td, p)
            t_metpy, _ = _metpy_lcl(p, t, td)
            diff       = abs(t_lcl - t_metpy)
            assert diff < tol, (
                f"Tropical: Bolton={t_lcl:.4f} K  MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )

    def test_array_elementwise_vs_metpy(self):
        """Vectorised array call must match MetPy per element."""
        temp_arr     = np.array([c[2] for c in STANDARD_CASES])
        dewpoint_arr = np.array([c[3] for c in STANDARD_CASES])
        pressure_arr = np.array([c[1] for c in STANDARD_CASES])

        t_arr, _ = Lcl.get_lcl_using_approximation(temp_arr, dewpoint_arr, pressure_arr)

        for i, (label, pressure, temp_k, dewpoint_k) in enumerate(STANDARD_CASES):
            depression = temp_k - dewpoint_k
            tol        = _metpy_tol(depression)
            t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
            diff       = abs(float(t_arr[i]) - t_metpy)
            assert diff < tol, (
                f"[{label}] Array[{i}]: Bolton={t_arr[i]:.4f} K  "
                f"MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )


# ===========================================================================
# Section 5 — Monotonicity
# ===========================================================================

class TestMonotonicity:

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_temp(self, base_temp, depressions):
        """Increasing dewpoint depression must lower T_LCL."""
        t_prev = None
        for dep in sorted(depressions):
            t_lcl, _ = _approx(base_temp, base_temp - dep, 1013.25)
            if t_prev is not None:
                assert t_lcl < t_prev, (
                    f"T={base_temp}, dep={dep}: T_LCL={t_lcl:.4f} not < prev {t_prev:.4f}"
                )
            t_prev = t_lcl

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_pressure(self, base_temp, depressions):
        """Increasing dewpoint depression must lower P_LCL."""
        p_prev = None
        for dep in sorted(depressions):
            _, p_lcl = _approx(base_temp, base_temp - dep, 1013.25)
            if p_prev is not None:
                assert p_lcl < p_prev
            p_prev = p_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_warmer_parcel_gives_warmer_lcl_temp(self, depression):
        """At fixed depression, warmer parcel must give warmer T_LCL."""
        temps  = [278.15, 283.15, 288.15, 293.15, 298.15]
        t_prev = None
        for t in temps:
            t_lcl, _ = _approx(t, t - depression, 1013.25)
            if t_prev is not None:
                assert t_lcl > t_prev
            t_prev = t_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_higher_surface_pressure_gives_higher_lcl_pressure(self, depression):
        """At fixed T and Td, higher P_surface must give higher P_LCL."""
        t, td     = 293.15, 293.15 - depression
        pressures = [850.0, 900.0, 950.0, 1013.25]
        p_prev    = None
        for p in pressures:
            _, p_lcl = _approx(t, td, p)
            if p_prev is not None:
                assert p_lcl > p_prev - 1e-6
            p_prev = p_lcl


# ===========================================================================
# Section 6 — Convergence
# ===========================================================================

class TestConvergence:

    def test_100k_random_parcels_all_finite(self):
        rng          = np.random.default_rng(42)
        n            = 100_000
        temp_k       = rng.uniform(243.15, 313.15, n)
        dewpoint_k   = temp_k - rng.uniform(0.5, 20.0, n)
        pressure_hpa = rng.uniform(850.0, 1013.25, n)

        t_lcl, p_lcl = Lcl.get_lcl_using_approximation(temp_k, dewpoint_k, pressure_hpa)

        assert np.all(np.isfinite(t_lcl)), (
            f"{np.sum(~np.isfinite(t_lcl))} non-finite T_LCL values"
        )
        assert np.all(np.isfinite(p_lcl)), (
            f"{np.sum(~np.isfinite(p_lcl))} non-finite P_LCL values"
        )
        assert np.all(t_lcl <= temp_k + 1e-9), "T_LCL > T_surface for some parcels"
        assert np.all(p_lcl <= pressure_hpa + 1e-9), "P_LCL > P_surface for some parcels"

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_all_finite(self, base_temp, depressions):
        for dep in depressions:
            t_lcl, p_lcl = _approx(base_temp, base_temp - dep, 1013.25)
            assert math.isfinite(t_lcl)
            assert math.isfinite(p_lcl)

    def test_large_array_poisson_holds(self):
        """Poisson relation must hold for all elements of a large random array."""
        rng          = np.random.default_rng(7)
        n            = 10_000
        temp_k       = rng.uniform(280.0, 305.0, n)
        dewpoint_k   = temp_k - rng.uniform(1.0, 10.0, n)
        pressure_hpa = rng.uniform(900.0, 1013.25, n)

        t_lcl, p_lcl = Lcl.get_lcl_using_approximation(temp_k, dewpoint_k, pressure_hpa)
        expected_p   = pressure_hpa * (t_lcl / temp_k) ** (cpd / Rd)

        np.testing.assert_allclose(p_lcl, expected_p, atol=1e-4,
                                   err_msg="Poisson relation violated in large array")


# ===========================================================================
# Section 7 — Regression guard
# ===========================================================================

class TestRegression:
    """
    Hard-coded values seeded from actual ``Lcl.get_lcl_using_approximation``
    output with the Bolton (1980) equation.  Any implementation change that
    alters results will fail immediately.
    """

    CASES = [
        # (temp_k, dewpoint_k, pressure_hpa, exp_t,   exp_p,   tol_t, tol_p)
        (293.15, 285.15, 1013.25, 283.3482, 899.5646, 1e-3, 1e-2),
        (303.15, 298.15, 1000.0,  296.9371, 930.1001, 1e-3, 1e-2),
        (278.15, 268.15, 1013.25, 266.1099, 867.8913, 1e-3, 1e-2),
        (300.15, 290.15, 1000.0,  287.8506, 863.7975, 1e-3, 1e-2),
        (290.15, 289.15, 1013.25, 288.9156, 998.2457, 1e-3, 1e-2),
    ]

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_temp(self, temp_k, dewpoint_k, pressure_hpa,
                             exp_t, exp_p, tol_t, tol_p):
        t_lcl, _ = _approx(temp_k, dewpoint_k, pressure_hpa)
        assert abs(t_lcl - exp_t) < tol_t, (
            f"T={temp_k}, Td={dewpoint_k}: got {t_lcl:.6f} K, "
            f"expected {exp_t:.6f} K (Δ={abs(t_lcl-exp_t):.6f})"
        )

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_pressure(self, temp_k, dewpoint_k, pressure_hpa,
                                 exp_t, exp_p, tol_t, tol_p):
        _, p_lcl = _approx(temp_k, dewpoint_k, pressure_hpa)
        assert abs(p_lcl - exp_p) < tol_p, (
            f"T={temp_k}, Td={dewpoint_k}: got {p_lcl:.4f} hPa, "
            f"expected {exp_p:.4f} hPa (Δ={abs(p_lcl-exp_p):.4f})"
        )


# ===========================================================================
# Section 8 — Consistency with BoltonLclEquation
# ===========================================================================

class TestConsistencyWithDirectEquation:
    """
    ``Lcl.get_lcl_using_approximation`` must give bit-identical results to
    calling ``BoltonLclEquation.calculate`` directly.  Validates that the
    public API adds no numerical distortion through the registry dispatch layer.
    """

    @pytest.fixture(scope="class")
    def bolton_eq(self):
        return BoltonLclEquation()

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_scalar_matches_direct_equation(self, bolton_eq, label, pressure, temp_k, dewpoint_k):
        t_pub, p_pub = _approx(temp_k, dewpoint_k, pressure)
        t_dir, p_dir = bolton_eq.calculate(temp_k, dewpoint_k, pressure)
        assert t_pub == t_dir, (
            f"[{label}] T: public={t_pub:.8f} K  direct={t_dir:.8f} K"
        )
        assert p_pub == p_dir, (
            f"[{label}] P: public={p_pub:.8f} hPa  direct={p_dir:.8f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_array_matches_direct_equation(self, bolton_eq, label, pressure, temp_k, dewpoint_k):
        t_a  = np.array([temp_k])
        td_a = np.array([dewpoint_k])
        p_a  = np.array([pressure])

        t_pub, p_pub = Lcl.get_lcl_using_approximation(t_a, td_a, p_a)
        t_dir, p_dir = bolton_eq.calculate(t_a, td_a, p_a)

        np.testing.assert_array_equal(t_pub, t_dir)
        np.testing.assert_array_equal(p_pub, p_dir)

    def test_full_standard_array_matches_direct(self, bolton_eq):
        """All standard cases as one array call must match direct equation."""
        temp_arr     = np.array([c[2] for c in STANDARD_CASES])
        dewpoint_arr = np.array([c[3] for c in STANDARD_CASES])
        pressure_arr = np.array([c[1] for c in STANDARD_CASES])

        t_pub, p_pub = Lcl.get_lcl_using_approximation(temp_arr, dewpoint_arr, pressure_arr)
        t_dir, p_dir = bolton_eq.calculate(temp_arr, dewpoint_arr, pressure_arr)

        np.testing.assert_array_equal(t_pub, t_dir)
        np.testing.assert_array_equal(p_pub, p_dir)


# ===========================================================================
# Section 9 — Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_very_small_depression(self):
        """0.5 K depression — nearly saturated parcel."""
        t_lcl, p_lcl = _approx(293.15, 292.65, 1013.25)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 293.15

    def test_high_surface_pressure(self):
        t_lcl, p_lcl = _approx(293.15, 285.15, 1050.0)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 1050.0

    def test_low_surface_pressure_high_elevation(self):
        t_lcl, p_lcl = _approx(285.15, 278.15, 850.0)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 850.0

    def test_tropical_warm_humid(self):
        t_lcl, p_lcl = _approx(308.15, 303.15, 1005.0)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 308.15

    def test_sensitivity_to_small_dewpoint_change(self):
        """A 0.01 K increase in T_d must raise T_LCL."""
        delta = 0.01
        t, td, p = 293.15, 285.15, 1013.25
        t1, _ = _approx(t, td,         p)
        t2, _ = _approx(t, td + delta, p)
        assert t2 > t1
        assert abs(t2 - t1) < delta * 10

    def test_large_depression_still_physical(self):
        """20 K dewpoint depression must still produce a valid LCL."""
        t_lcl, p_lcl = _approx(300.15, 280.15, 1013.25)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 300.15
        assert p_lcl <= 1013.25

    def test_array_mixed_pressure_levels(self):
        """Array with varying pressure levels must produce physical results."""
        temp_arr     = np.array([293.15, 288.15, 303.15, 278.15])
        dewpoint_arr = np.array([285.15, 283.15, 298.15, 273.15])
        pressure_arr = np.array([1013.25, 950.0, 1000.0, 900.0])

        t_lcl, p_lcl = Lcl.get_lcl_using_approximation(temp_arr, dewpoint_arr, pressure_arr)

        assert np.all(np.isfinite(t_lcl))
        assert np.all(np.isfinite(p_lcl))
        assert np.all(t_lcl <= temp_arr + 1e-9)
        assert np.all(p_lcl <= pressure_arr + 1e-9)

    def test_single_element_array_not_scalar(self):
        """Single-element array input must return ndarray, not scalar."""
        t, p = Lcl.get_lcl_using_approximation(
            np.array([293.15]), np.array([285.15]), np.array([1013.25])
        )
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])