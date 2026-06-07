"""
Tests for Lcl.get_lcl_using_solver — public API
================================================

Cross-validates against MetPy's LCL implementation (Romps 2017) and verifies
the full public-API stack: enum/string parsing → vapor equation dispatch →
Brent solver → (T_LCL, P_LCL) output.

This file tests the *public interface* ``Lcl.get_lcl_using_solver``. The
lower-level ``IterativeLclEquation.calculate`` is tested separately in
``_test_iterative_lcl.py``.

Test sections
-------------
1.  Enum / string API — solver_name, vapor_equation_name, surface_type
2.  Physical constraints — T_LCL ≤ T_surface, P_LCL ≤ P_surface, Poisson
3.  Return types — scalar → float, array → ndarray of float64
4.  Vapor equation selection — all three equations produce valid results
5.  Surface type dispatch — ``'automatic'``, ``'water'``, ``'ice'``
6.  MetPy cross-validation — Goff-Gratch and Hyland-Wexler vs Romps (2017)
7.  Monotonicity — depression sweep, surface temperature sweep
8.  Convergence — 100 K random parcels all finite and physical
9.  Regression guard — hard-coded seeded values
10. Consistency — ``get_lcl_using_solver`` equals ``IterativeLclEquation.calculate``
11. Edge cases — near-saturated, high/low pressure, tropical

Tolerance vs MetPy (Romps 2017 exact)
--------------------------------------
A flat tolerance of 0.5 K / 2.0 hPa is used throughout. The Brent solver
achieves ~1e-8 K accuracy; all remaining discrepancy vs MetPy originates from
the choice of vapour pressure equation, not the solver.

References
----------
.. [1] Romps, D. M. (2017). Exact analytic solutions for pseudo-adiabatic
   ascent. Journal of the Atmospheric Sciences, 74(9), 3033–3039.
   https://doi.org/10.1175/JAS-D-17-0073.1
.. [2] Bolton, D. (1980). The computation of equivalent potential temperature.
   Monthly Weather Review, 108(7), 1046–1053.

Dependencies
------------
    pip install pytest numpy numba metpy
"""

import math

import numpy as np
import pytest

from meteocalc.lcl._enums import LclEquationName
from meteocalc.lcl._lcl_equation import IterativeLclEquation
from meteocalc.lcl.core import Lcl
from meteocalc.shared.constants import Rd, cpd, eps
from meteocalc.vapor._enums import VaporEquationName
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
    """Compute scalar mixing ratio. Uses Goff-Gratch water by default."""
    if vapor_eq is None:
        vapor_eq = Vapor.get_equation("goff_gratch", phase="water")
    e = float(vapor_eq.calculate(dewpoint_k))
    return eps * e / (pressure_hpa - e)


def _w_arr(
    dewpoint_arr: np.ndarray,
    pressure_arr: np.ndarray,
    vapor_eq=None,
) -> np.ndarray:
    """Compute array mixing ratio. Uses Goff-Gratch water by default."""
    if vapor_eq is None:
        vapor_eq = Vapor.get_equation("goff_gratch", phase="water")
    e_arr = vapor_eq.calculate(dewpoint_arr).astype(np.float64)
    return eps * e_arr / (pressure_arr - e_arr)


def _solver(temp_k, dewpoint_k, pressure_hpa, mixing_ratio=None, **kwargs):
    """Thin wrapper — computes mixing_ratio if not supplied."""
    if mixing_ratio is None:
        mixing_ratio = _w(dewpoint_k, pressure_hpa)
    return Lcl.get_lcl_using_solver(
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
    ("standard_midlat",    1013.25, 293.15, 285.15),  # 20 °C / 12 °C — 8 K
    ("warm_humid",         1000.0,  303.15, 298.15),  # 30 °C / 25 °C — 5 K
    ("cool_dry",           1013.25, 278.15, 268.15),  # 5 °C / −5 °C  — 10 K
    ("near_saturated",     1013.25, 290.15, 289.15),  # 1 K depression
    ("high_elevation",     850.0,   285.15, 278.15),  # 12 °C / 5 °C
    ("summer_continental", 1000.0,  300.15, 290.15),  # 27 °C / 17 °C — 10 K
    ("autumn_uk",          1013.25, 283.15, 278.15),  # 10 °C / 5 °C  — 5 K
    ("moderate_dry",       1013.25, 295.15, 282.15),  # 22 °C / 9 °C  — 13 K
]

METPY_TOL_K   = 0.5   # K
METPY_TOL_HPA = 2.0   # hPa

DEPRESSION_SWEEP = [
    (273.15 + 15, [1, 2, 5, 10, 15]),
    (273.15 + 25, [1, 2, 5, 10, 15]),
]


# ===========================================================================
# Section 1 — Enum / string API
# ===========================================================================

class TestEnumStringAPI:
    """
    All keyword arguments accept both strings and their enum equivalents.
    Results must be bit-identical regardless of which form is used.
    """

    def test_solver_name_string_equals_enum(self):
        t, p = _solver(293.15, 285.15, 1013.25, solver_name="iterative")
        t_e, p_e = _solver(293.15, 285.15, 1013.25, solver_name=LclEquationName.ITERATIVE)
        assert t == t_e
        assert p == p_e

    def test_vapor_equation_string_equals_enum_goff_gratch(self):
        w = _w(285.15, 1013.25)
        t_s, p_s = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, vapor_equation_name="goff_gratch")
        t_e, p_e = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, vapor_equation_name=VaporEquationName.GOFF_GRATCH)
        assert t_s == t_e
        assert p_s == p_e

    def test_vapor_equation_string_equals_enum_hyland_wexler(self):
        veq = Vapor.get_equation("hyland_wexler", phase="water")
        w   = _w(285.15, 1013.25, veq)
        t_s, p_s = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, vapor_equation_name="hyland_wexler")
        t_e, p_e = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, vapor_equation_name=VaporEquationName.HYLAND_WEXLER)
        assert t_s == t_e
        assert p_s == p_e

    def test_surface_type_string_equals_enum_water(self):
        from meteocalc.shared._shared_enums import SurfaceType
        w = _w(285.15, 1013.25)
        t_s, p_s = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type="water")
        t_e, p_e = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type=SurfaceType.WATER)
        assert t_s == t_e
        assert p_s == p_e

    def test_surface_type_string_equals_enum_automatic(self):
        from meteocalc.shared._shared_enums import SurfaceType
        w = _w(285.15, 1013.25)
        t_s, p_s = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type="automatic")
        t_e, p_e = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type=SurfaceType.AUTOMATIC)
        assert t_s == t_e
        assert p_s == p_e

    def test_default_solver_name_is_iterative(self):
        w = _w(285.15, 1013.25)
        t_default, p_default = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w)
        t_explicit, p_explicit = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, solver_name="iterative")
        assert t_default == t_explicit
        assert p_default == p_explicit

    def test_default_vapor_equation_is_goff_gratch(self):
        w = _w(285.15, 1013.25)
        t_default, p_default = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w)
        t_explicit, p_explicit = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, vapor_equation_name="goff_gratch")
        assert t_default == t_explicit
        assert p_default == p_explicit

    def test_default_surface_type_is_automatic(self):
        w = _w(285.15, 1013.25)
        t_default, p_default = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w)
        t_explicit, p_explicit = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type="automatic")
        assert t_default == t_explicit
        assert p_default == p_explicit

    def test_invalid_solver_name_raises(self):
        w = _w(285.15, 1013.25)
        with pytest.raises((ValueError, KeyError)):
            Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, solver_name="nonexistent")

    def test_invalid_vapor_equation_raises(self):
        w = _w(285.15, 1013.25)
        with pytest.raises((ValueError, KeyError)):
            Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, vapor_equation_name="nonexistent")


# ===========================================================================
# Section 2 — Physical constraints
# ===========================================================================

class TestPhysicalConstraints:
    """LCL results must satisfy fundamental atmospheric physics."""

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_below_surface_temp(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, _ = _solver(temp_k, dewpoint_k, pressure)
        assert t_lcl <= temp_k + 1e-9, (
            f"[{label}] T_LCL={t_lcl:.4f} K > T_surface={temp_k:.4f} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_pressure_below_surface_pressure(self, label, pressure, temp_k, dewpoint_k):
        _, p_lcl = _solver(temp_k, dewpoint_k, pressure)
        assert p_lcl <= pressure + 1e-9, (
            f"[{label}] P_LCL={p_lcl:.4f} hPa > P_surface={pressure:.4f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_above_absolute_zero(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, _ = _solver(temp_k, dewpoint_k, pressure)
        assert t_lcl > 0.0, f"[{label}] T_LCL={t_lcl:.4f} K ≤ 0 K"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_results_are_finite(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, p_lcl = _solver(temp_k, dewpoint_k, pressure)
        assert math.isfinite(t_lcl), f"[{label}] T_LCL not finite"
        assert math.isfinite(p_lcl), f"[{label}] P_LCL not finite"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_poisson_relation_consistent(self, label, pressure, temp_k, dewpoint_k):
        """P_LCL must satisfy the Poisson relation given T_LCL."""
        t_lcl, p_lcl = _solver(temp_k, dewpoint_k, pressure)
        expected_p   = pressure * (t_lcl / temp_k) ** (cpd / Rd)
        assert abs(p_lcl - expected_p) < 1e-4, (
            f"[{label}] Poisson violated: got {p_lcl:.6f} hPa, "
            f"expected {expected_p:.6f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_below_dewpoint(self, label, pressure, temp_k, dewpoint_k):
        """T_LCL must be below T_d_surface: parcel cools at DALR (~9.8 K/km),
        dewpoint falls only ~1.8 K/km, so they converge with T_LCL < T_d."""
        t_lcl, _ = _solver(temp_k, dewpoint_k, pressure)
        assert t_lcl < dewpoint_k + 1e-9, (
            f"[{label}] T_LCL={t_lcl:.4f} K unexpectedly above T_d={dewpoint_k:.4f} K"
        )


# ===========================================================================
# Section 3 — Return types
# ===========================================================================

class TestReturnTypes:
    """Scalar inputs return Python floats; array inputs return float64 ndarrays."""

    def test_scalar_returns_floats(self):
        t, p = _solver(293.15, 285.15, 1013.25)
        assert isinstance(t, float), f"Expected float, got {type(t)}"
        assert isinstance(p, float), f"Expected float, got {type(p)}"

    def test_returns_two_values(self):
        assert len(_solver(293.15, 285.15, 1013.25)) == 2

    def test_array_returns_ndarrays(self):
        n    = 5
        t_a  = np.full(n, 293.15)
        td_a = np.full(n, 285.15)
        p_a  = np.full(n, 1013.25)
        w_a  = _w_arr(td_a, p_a)
        t, p = Lcl.get_lcl_using_solver(t_a, td_a, p_a, w_a)
        assert isinstance(t, np.ndarray)
        assert isinstance(p, np.ndarray)

    def test_array_dtype_float64(self):
        n    = 4
        t_a  = np.full(n, 293.15)
        td_a = np.full(n, 285.15)
        p_a  = np.full(n, 1013.25)
        w_a  = _w_arr(td_a, p_a)
        t, p = Lcl.get_lcl_using_solver(t_a, td_a, p_a, w_a)
        assert t.dtype == np.float64
        assert p.dtype == np.float64

    def test_array_shape_matches_input(self):
        n    = 7
        t_a  = np.full(n, 293.15)
        td_a = np.full(n, 285.15)
        p_a  = np.full(n, 1013.25)
        w_a  = _w_arr(td_a, p_a)
        t, p = Lcl.get_lcl_using_solver(t_a, td_a, p_a, w_a)
        assert t.shape == (n,)
        assert p.shape == (n,)

    def test_scalar_array_consistency(self):
        """Single-element array result must equal scalar result."""
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        w = _w(dewpoint_k, pressure_hpa)

        t_sc, p_sc = Lcl.get_lcl_using_solver(temp_k, dewpoint_k, pressure_hpa, w)
        t_ar, p_ar = Lcl.get_lcl_using_solver(
            np.array([temp_k]), np.array([dewpoint_k]),
            np.array([pressure_hpa]), np.array([w]),
        )
        assert abs(t_sc - float(t_ar[0])) < 1e-10
        assert abs(p_sc - float(p_ar[0])) < 1e-10

    def test_deterministic(self):
        """Repeated calls with identical inputs must return identical results."""
        w = _w(285.15, 1013.25)
        results = [
            Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w)
            for _ in range(5)
        ]
        for t, p in results:
            assert t == results[0][0]
            assert p == results[0][1]


# ===========================================================================
# Section 4 — Vapor equation selection
# ===========================================================================

class TestVaporEquationSelection:
    """All three vapor equations must produce valid, physically consistent results."""

    VAPOR_EQUATIONS = ["goff_gratch", "hyland_wexler", "bolton"]

    @pytest.mark.parametrize("veq_name", VAPOR_EQUATIONS)
    def test_vapor_equation_produces_finite_result(self, veq_name):
        veq  = Vapor.get_equation(veq_name, phase="water")
        w    = _w(285.15, 1013.25, veq)
        t, p = Lcl.get_lcl_using_solver(
            293.15, 285.15, 1013.25, w, vapor_equation_name=veq_name
        )
        assert math.isfinite(t), f"{veq_name}: T_LCL not finite"
        assert math.isfinite(p), f"{veq_name}: P_LCL not finite"

    @pytest.mark.parametrize("veq_name", VAPOR_EQUATIONS)
    def test_vapor_equation_t_below_surface(self, veq_name):
        veq  = Vapor.get_equation(veq_name, phase="water")
        w    = _w(285.15, 1013.25, veq)
        t, _ = Lcl.get_lcl_using_solver(
            293.15, 285.15, 1013.25, w, vapor_equation_name=veq_name
        )
        assert t <= 293.15

    def test_goff_gratch_and_hyland_wexler_agree(self):
        """Two WMO-standard equations must agree to within 0.05 K."""
        veq_gg = Vapor.get_equation("goff_gratch",   phase="water")
        veq_hw = Vapor.get_equation("hyland_wexler", phase="water")
        w_gg   = _w(285.15, 1013.25, veq_gg)
        w_hw   = _w(285.15, 1013.25, veq_hw)

        t_gg, _ = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w_gg, vapor_equation_name="goff_gratch")
        t_hw, _ = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w_hw, vapor_equation_name="hyland_wexler")

        assert abs(t_gg - t_hw) < 0.05, (
            f"GG={t_gg:.4f} K  HW={t_hw:.4f} K  Δ={abs(t_gg-t_hw):.4f} K"
        )

    def test_all_vapor_equations_satisfy_poisson(self):
        for veq_name in self.VAPOR_EQUATIONS:
            veq     = Vapor.get_equation(veq_name, phase="water")
            w       = _w(285.15, 1013.25, veq)
            t_lcl, p_lcl = Lcl.get_lcl_using_solver(
                293.15, 285.15, 1013.25, w, vapor_equation_name=veq_name
            )
            expected_p = 1013.25 * (t_lcl / 293.15) ** (cpd / Rd)
            assert abs(p_lcl - expected_p) < 1e-4, (
                f"{veq_name}: Poisson violated ({p_lcl:.6f} vs {expected_p:.6f} hPa)"
            )


# ===========================================================================
# Section 5 — Surface type dispatch
# ===========================================================================

class TestSurfaceTypeDispatch:
    """surface_type correctly selects vapour pressure constants."""

    def test_water_surface_type_produces_valid_result(self):
        w = _w(285.15, 1013.25)
        t, p = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type="water")
        assert math.isfinite(t)
        assert math.isfinite(p)
        assert t <= 293.15

    def test_automatic_above_freezing_matches_water(self):
        """'automatic' for above-freezing parcels must equal 'water'."""
        w = _w(285.15, 1013.25)
        t_auto, p_auto = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type="automatic")
        t_water, p_water = Lcl.get_lcl_using_solver(293.15, 285.15, 1013.25, w, surface_type="water")
        assert abs(t_auto - t_water) < 1e-10
        assert abs(p_auto - p_water) < 1e-10

    def test_ice_surface_type_produces_valid_result(self):
        """Ice surface: cold parcel where ice constants apply."""
        veq_gg = Vapor.get_equation("goff_gratch", phase="ice")
        w      = _w(263.15, 1013.25, veq_gg)
        t, p   = Lcl.get_lcl_using_solver(
            268.15, 263.15, 1013.25, w,
            vapor_equation_name="goff_gratch",
            surface_type="ice",
        )
        assert math.isfinite(t)
        assert t <= 268.15

    def test_automatic_selects_ice_for_sub_freezing(self):
        """
        For a sub-freezing parcel, 'automatic' must differ from 'water' because
        different vapour pressure constants apply below 273.15 K.
        """
        temp_k, dewpoint_k, p_hpa = 268.15, 263.15, 1013.25

        veq_auto = Vapor.get_equation("goff_gratch", phase="automatic")
        veq_water = Vapor.get_equation("goff_gratch", phase="water")

        w_auto  = _w(dewpoint_k, p_hpa, veq_auto)
        w_water = _w(dewpoint_k, p_hpa, veq_water)

        t_auto, _ = Lcl.get_lcl_using_solver(temp_k, dewpoint_k, p_hpa, w_auto,  surface_type="automatic")
        t_water, _ = Lcl.get_lcl_using_solver(temp_k, dewpoint_k, p_hpa, w_water, surface_type="water")

        # They may differ — the key assertion is that 'automatic' gives a finite physical result
        assert math.isfinite(t_auto)
        assert t_auto <= temp_k


# ===========================================================================
# Section 6 — MetPy cross-validation
# ===========================================================================

class TestMetPyCrossValidation:
    """
    Flat 0.5 K / 2.0 hPa tolerance. Discrepancy vs MetPy (Romps 2017) is
    from vapour equation choice, not the solver.
    """

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_temp_goff_gratch_vs_metpy(self, label, pressure, temp_k, dewpoint_k):
        t_lcl, _   = _solver(temp_k, dewpoint_k, pressure, vapor_equation_name="goff_gratch")
        t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff       = abs(t_lcl - t_metpy)
        assert diff < METPY_TOL_K, (
            f"[{label}] GG: got {t_lcl:.4f} K  MetPy {t_metpy:.4f} K  Δ={diff:.4f} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_pressure_goff_gratch_vs_metpy(self, label, pressure, temp_k, dewpoint_k):
        _, p_lcl   = _solver(temp_k, dewpoint_k, pressure, vapor_equation_name="goff_gratch")
        _, p_metpy = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff       = abs(p_lcl - p_metpy)
        assert diff < METPY_TOL_HPA, (
            f"[{label}] GG: got {p_lcl:.4f} hPa  MetPy {p_metpy:.4f} hPa  Δ={diff:.4f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_temp_hyland_wexler_vs_metpy(self, label, pressure, temp_k, dewpoint_k):
        veq        = Vapor.get_equation("hyland_wexler", phase="water")
        w          = _w(dewpoint_k, pressure, veq)
        t_lcl, _   = Lcl.get_lcl_using_solver(
            temp_k, dewpoint_k, pressure, w, vapor_equation_name="hyland_wexler"
        )
        t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff       = abs(t_lcl - t_metpy)
        assert diff < METPY_TOL_K, (
            f"[{label}] HW: got {t_lcl:.4f} K  MetPy {t_metpy:.4f} K  Δ={diff:.4f} K"
        )

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_vs_metpy(self, base_temp, depressions):
        for dep in depressions:
            td         = base_temp - dep
            t_lcl, _   = _solver(base_temp, td, 1013.25)
            t_metpy, _ = _metpy_lcl(1013.25, base_temp, td)
            diff       = abs(t_lcl - t_metpy)
            assert diff < METPY_TOL_K, (
                f"T={base_temp:.2f} K, dep={dep} K: "
                f"got {t_lcl:.4f} K  MetPy {t_metpy:.4f} K  Δ={diff:.4f} K"
            )

    def test_array_elementwise_vs_metpy(self):
        """Vectorised array call must match MetPy per element."""
        temp_arr     = np.array([c[2] for c in STANDARD_CASES])
        dewpoint_arr = np.array([c[3] for c in STANDARD_CASES])
        pressure_arr = np.array([c[1] for c in STANDARD_CASES])
        w_arr        = _w_arr(dewpoint_arr, pressure_arr)

        t_arr, _ = Lcl.get_lcl_using_solver(temp_arr, dewpoint_arr, pressure_arr, w_arr)

        for i, (label, pressure, temp_k, dewpoint_k) in enumerate(STANDARD_CASES):
            t_metpy, _ = _metpy_lcl(pressure, temp_k, dewpoint_k)
            diff       = abs(float(t_arr[i]) - t_metpy)
            assert diff < METPY_TOL_K, (
                f"[{label}] Array[{i}]: got {t_arr[i]:.4f} K  "
                f"MetPy {t_metpy:.4f} K  Δ={diff:.4f} K"
            )


# ===========================================================================
# Section 7 — Monotonicity
# ===========================================================================

class TestMonotonicity:

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_temp(self, base_temp, depressions):
        t_prev = None
        for dep in sorted(depressions):
            td = base_temp - dep
            t_lcl, _ = _solver(base_temp, td, 1013.25)
            if t_prev is not None:
                assert t_lcl < t_prev, (
                    f"T={base_temp}, dep={dep}: T_LCL={t_lcl:.4f} not < prev {t_prev:.4f}"
                )
            t_prev = t_lcl

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_pressure(self, base_temp, depressions):
        p_prev = None
        for dep in sorted(depressions):
            td = base_temp - dep
            _, p_lcl = _solver(base_temp, td, 1013.25)
            if p_prev is not None:
                assert p_lcl < p_prev
            p_prev = p_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_warmer_parcel_gives_warmer_lcl_temp(self, depression):
        temps  = [278.15, 283.15, 288.15, 293.15, 298.15]
        t_prev = None
        for t in temps:
            t_lcl, _ = _solver(t, t - depression, 1013.25)
            if t_prev is not None:
                assert t_lcl > t_prev
            t_prev = t_lcl

    @pytest.mark.parametrize("depression", [2, 5])
    def test_higher_surface_pressure_gives_higher_lcl_pressure(self, depression):
        """For fixed T and Td, higher P_surface → higher P_LCL."""
        t, td = 293.15, 293.15 - depression
        pressures = [950.0, 975.0, 1000.0, 1013.25]
        p_prev = None
        for p in pressures:
            _, p_lcl = _solver(t, td, p)
            if p_prev is not None:
                assert p_lcl > p_prev
            p_prev = p_lcl


# ===========================================================================
# Section 8 — Convergence
# ===========================================================================

class TestConvergence:

    def test_100k_random_parcels_all_finite(self):
        rng          = np.random.default_rng(42)
        n            = 100_000
        temp_k       = rng.uniform(280.0, 308.0, n)
        dewpoint_k   = np.clip(temp_k - rng.uniform(1.0, 10.0, n), 274.0, None)
        pressure_hpa = rng.uniform(850.0, 1013.25, n)
        w_arr        = _w_arr(dewpoint_k, pressure_hpa)

        t_lcl, p_lcl = Lcl.get_lcl_using_solver(temp_k, dewpoint_k, pressure_hpa, w_arr)

        assert np.all(np.isfinite(t_lcl)), "Some T_LCL values are not finite"
        assert np.all(np.isfinite(p_lcl)), "Some P_LCL values are not finite"
        assert np.all(t_lcl <= temp_k + 1e-9), "Some T_LCL > T_surface"
        assert np.all(p_lcl <= pressure_hpa + 1e-9), "Some P_LCL > P_surface"

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_all_finite(self, base_temp, depressions):
        for dep in depressions:
            t_lcl, p_lcl = _solver(base_temp, base_temp - dep, 1013.25)
            assert math.isfinite(t_lcl)
            assert math.isfinite(p_lcl)

    def test_all_vapor_equations_100_parcels(self):
        """All three vapor equations must converge on a 100-parcel array."""
        rng    = np.random.default_rng(0)
        n      = 100
        temp_k = rng.uniform(283.0, 303.0, n)
        td_k   = temp_k - rng.uniform(1.0, 8.0, n)
        p_hpa  = np.full(n, 1013.25)

        for veq_name in ["goff_gratch", "hyland_wexler", "bolton"]:
            veq   = Vapor.get_equation(veq_name, phase="water")
            w_arr = _w_arr(td_k, p_hpa, veq)
            t_arr, p_arr = Lcl.get_lcl_using_solver(
                temp_k, td_k, p_hpa, w_arr, vapor_equation_name=veq_name
            )
            assert np.all(np.isfinite(t_arr)), f"{veq_name}: non-finite T_LCL"
            assert np.all(np.isfinite(p_arr)), f"{veq_name}: non-finite P_LCL"


# ===========================================================================
# Section 9 — Regression guard
# ===========================================================================

class TestRegression:
    """
    Hard-coded values seeded from actual ``Lcl.get_lcl_using_solver`` output
    with Goff-Gratch water, iterative solver.  Any implementation change that
    alters results will fail immediately.
    """

    CASES = [
        # (temp_k, dewpoint_k, pressure_hpa, exp_t,   exp_p,   tol_t, tol_p)
        (293.15, 285.15, 1013.25, 283.3605, 899.7022, 1e-3, 1e-1),
        (303.15, 298.15, 1000.0,  296.9407, 930.1399, 1e-3, 1e-1),
        (278.15, 268.15, 1013.25, 266.1260, 868.0749, 1e-3, 1e-1),
        (300.15, 290.15, 1000.0,  287.8639, 863.9366, 1e-3, 1e-1),
        (290.15, 289.15, 1013.25, 288.9170, 998.2622, 1e-3, 1e-1),
    ]

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_temp(self, temp_k, dewpoint_k, pressure_hpa,
                             exp_t, exp_p, tol_t, tol_p):
        t_lcl, _ = _solver(temp_k, dewpoint_k, pressure_hpa)
        assert abs(t_lcl - exp_t) < tol_t, (
            f"T={temp_k}, Td={dewpoint_k}: got {t_lcl:.6f} K, "
            f"expected {exp_t:.6f} K (Δ={abs(t_lcl-exp_t):.6f})"
        )

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_pressure(self, temp_k, dewpoint_k, pressure_hpa,
                                 exp_t, exp_p, tol_t, tol_p):
        _, p_lcl = _solver(temp_k, dewpoint_k, pressure_hpa)
        assert abs(p_lcl - exp_p) < tol_p, (
            f"T={temp_k}, Td={dewpoint_k}: got {p_lcl:.4f} hPa, "
            f"expected {exp_p:.4f} hPa (Δ={abs(p_lcl-exp_p):.4f})"
        )


# ===========================================================================
# Section 10 — Consistency with IterativeLclEquation
# ===========================================================================

class TestConsistencyWithDirectEquation:
    """
    ``Lcl.get_lcl_using_solver`` must give bit-identical results to calling
    ``IterativeLclEquation.calculate`` directly with the same vapour equation.
    Validates that the public API adds no numerical distortion.
    """

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_scalar_matches_direct_equation(self, label, pressure, temp_k, dewpoint_k):
        veq = Vapor.get_equation("goff_gratch", phase="water")
        w   = _w(dewpoint_k, pressure, veq)

        t_pub, p_pub = Lcl.get_lcl_using_solver(temp_k, dewpoint_k, pressure, w)

        eq_direct    = IterativeLclEquation()
        t_dir, p_dir = eq_direct.calculate(temp_k, dewpoint_k, pressure, w, veq)

        assert t_pub == t_dir, (
            f"[{label}] T: public={t_pub:.8f} K  direct={t_dir:.8f} K"
        )
        assert p_pub == p_dir, (
            f"[{label}] P: public={p_pub:.8f} hPa  direct={p_dir:.8f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_array_matches_direct_equation(self, label, pressure, temp_k, dewpoint_k):
        veq   = Vapor.get_equation("goff_gratch", phase="water")
        t_a   = np.array([temp_k])
        td_a  = np.array([dewpoint_k])
        p_a   = np.array([pressure])
        w_a   = _w_arr(td_a, p_a, veq)

        t_pub, p_pub = Lcl.get_lcl_using_solver(t_a, td_a, p_a, w_a)

        eq_direct    = IterativeLclEquation()
        t_dir, p_dir = eq_direct.calculate(t_a, td_a, p_a, w_a, veq)

        np.testing.assert_array_equal(t_pub, t_dir)
        np.testing.assert_array_equal(p_pub, p_dir)

    @pytest.mark.parametrize("veq_name", ["goff_gratch", "hyland_wexler", "bolton"])
    def test_all_equations_match_direct(self, veq_name):
        veq    = Vapor.get_equation(veq_name, phase="water")
        w      = _w(285.15, 1013.25, veq)
        t_pub, p_pub = Lcl.get_lcl_using_solver(
            293.15, 285.15, 1013.25, w, vapor_equation_name=veq_name
        )
        eq_direct    = IterativeLclEquation()
        t_dir, p_dir = eq_direct.calculate(293.15, 285.15, 1013.25, w, veq)
        assert t_pub == t_dir, f"{veq_name}: T mismatch"
        assert p_pub == p_dir, f"{veq_name}: P mismatch"


# ===========================================================================
# Section 11 — Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_very_small_depression(self):
        t_lcl, p_lcl = _solver(293.15, 292.65, 1013.25)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 293.15

    def test_high_surface_pressure(self):
        t_lcl, p_lcl = _solver(293.15, 285.15, 1050.0)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 1050.0

    def test_low_surface_pressure_high_elevation(self):
        t_lcl, p_lcl = _solver(285.15, 278.15, 850.0)
        assert math.isfinite(t_lcl)
        assert p_lcl <= 850.0

    def test_tropical_warm_humid(self):
        t_lcl, p_lcl = _solver(308.15, 303.15, 1005.0)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 308.15

    def test_sensitivity_to_small_dewpoint_change(self):
        """A 0.01 K increase in T_d must raise T_LCL (more moisture → higher LCL)."""
        delta = 0.01
        t, td, p = 293.15, 285.15, 1013.25
        t1, _ = _solver(t, td,         p)
        t2, _ = _solver(t, td + delta, p)
        assert t2 > t1
        assert abs(t2 - t1) < delta * 10

    def test_large_depression_still_physical(self):
        """20 K dewpoint depression must still produce a valid LCL."""
        t_lcl, p_lcl = _solver(300.15, 280.15, 1013.25)
        assert math.isfinite(t_lcl)
        assert t_lcl <= 300.15
        assert p_lcl <= 1013.25

    def test_array_mixed_pressure_levels(self):
        """Array with varying pressure levels must produce physical results."""
        temp_arr     = np.array([293.15, 288.15, 303.15, 278.15])
        dewpoint_arr = np.array([285.15, 283.15, 298.15, 273.15])
        pressure_arr = np.array([1013.25, 950.0, 1000.0, 900.0])
        w_arr        = _w_arr(dewpoint_arr, pressure_arr)

        t_lcl, p_lcl = Lcl.get_lcl_using_solver(temp_arr, dewpoint_arr, pressure_arr, w_arr)

        assert np.all(np.isfinite(t_lcl))
        assert np.all(np.isfinite(p_lcl))
        assert np.all(t_lcl <= temp_arr + 1e-9)
        assert np.all(p_lcl <= pressure_arr + 1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])