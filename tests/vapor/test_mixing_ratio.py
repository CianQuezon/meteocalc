"""
Tests for ``Vapor.get_mixing_ratio``
====================================

Mixing ratio w = ε · e / (p − e) in kg/kg, from vapour pressure ``e`` and
total pressure ``p`` (both hPa).

Verification strategy
---------------------
* **Closed form** — an independent pure-Python reference using the same ε,
  so any deviation beyond float64 round-off is a bug.
* **MetPy** (``mixing_ratio``, ``vapor_pressure``, ``saturation_mixing_ratio``)
  and **PsychroLib / ASHRAE 2017** (``GetHumRatioFromVapPres``,
  ``GetVapPresFromHumRatio``, ``GetSatHumRatio``) as external references.
  Both use a slightly different ε, which bounds the expected disagreement.
* **Physical invariants** — monotonicity, scale invariance, limits.
* **API contract** — scalar/array types, shapes, broadcasting, NaN guard.

Test sections
-------------
1.  Reference values      — closed form, hand-checked values, inverse
2.  Physical properties   — sign, monotonicity, scaling, limits
3.  Unphysical input      — p <= e returns NaN element-wise
4.  Input handling        — types, shapes, broadcasting, immutability
5.  Accuracy vs MetPy
6.  Accuracy vs PsychroLib (ASHRAE)
7.  Saturation mixing ratio via get_vapor_saturation
"""

import math

import numpy as np
import pytest

from meteocalc.shared.constants import eps
from meteocalc.vapor.core import Vapor

try:
    from metpy.calc import mixing_ratio as mp_mixing_ratio
    from metpy.calc import saturation_mixing_ratio as mp_saturation_mixing_ratio
    from metpy.calc import vapor_pressure as mp_vapor_pressure
    from metpy.units import units as mp_units

    HAS_METPY = True
except ImportError:
    HAS_METPY = False

try:
    import psychrolib

    psychrolib.SetUnitSystem(psychrolib.SI)
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False

requires_metpy = pytest.mark.skipif(not HAS_METPY, reason="metpy not installed")
requires_psychrolib = pytest.mark.skipif(
    not HAS_PSYCHROLIB, reason="psychrolib not installed"
)


# ============================================================================
# Tolerances
# ============================================================================

RTOL_EXACT = 1e-12
"""Same formula and ε as the reference → float64 round-off only."""

RTOL_EPS = 1e-4
"""External libraries use a different ε: ours Rd/Rv = 0.621993,
MetPy Mw/Md = 0.621957, ASHRAE 0.621945. Measured max deviation 7.8e-5."""

RTOL_SATURATION = 5e-3
"""Saturation mixing ratio additionally depends on the es formula
(Goff-Gratch here vs Ambaum in MetPy, Hyland-Wexler in ASHRAE).
Measured max deviation 2.2e-3."""


# ============================================================================
# Test data
# ============================================================================

MOIST_CASES = [
    # (label, vapor_pressure_hpa, pressure_hpa)
    ("very_dry_surface", 0.5, 1000.0),
    ("temperate_surface", 12.0, 1013.25),
    ("humid_surface", 23.37, 1000.0),
    ("tropical_surface", 37.8, 1010.0),
    ("extreme_humid", 50.0, 1010.0),
    ("level_850hpa", 8.0, 850.0),
    ("level_700hpa", 3.0, 700.0),
    ("level_500hpa", 1.0, 500.0),
    ("level_300hpa", 0.3, 300.0),
    ("stratosphere", 0.005, 100.0),
]
CASE_IDS = [c[0] for c in MOIST_CASES]
E_CASES = np.array([c[1] for c in MOIST_CASES])
P_CASES = np.array([c[2] for c in MOIST_CASES])

SATURATION_CASES = [
    # (label, temp_c, pressure_hpa) — away from 0 °C so library-specific
    # water/ice switch-over points cannot cause a mismatch.
    ("hot", 35.0, 1010.0),
    ("warm", 20.0, 1000.0),
    ("mild", 10.0, 1013.25),
    ("cool_850", 5.0, 850.0),
    ("cold_700", -10.0, 700.0),
    ("cold_500", -25.0, 500.0),
]
SAT_IDS = [c[0] for c in SATURATION_CASES]


def ref_mixing_ratio(e, p):
    """Independent closed-form reference: w = ε e / (p − e)."""
    return eps * e / (p - e)


def ref_vapor_pressure(w, p):
    """Algebraic inverse: e = w p / (ε + w)."""
    return w * p / (eps + w)


@pytest.fixture
def random_inputs():
    """10 000 physically valid (e, p) pairs spanning the troposphere."""
    rng = np.random.default_rng(1234)
    p = rng.uniform(100.0, 1050.0, 10_000)
    e = rng.uniform(0.0, 0.06, 10_000) * p  # e/p ≤ 6 %
    return e, p


# ============================================================================
# 1. Reference values
# ============================================================================


class TestReferenceValues:
    @pytest.mark.parametrize("label, e, p", MOIST_CASES, ids=CASE_IDS)
    def test_matches_closed_form(self, label, e, p):
        got = Vapor.get_mixing_ratio(e, p)
        assert got == pytest.approx(ref_mixing_ratio(e, p), rel=RTOL_EXACT)

    @pytest.mark.parametrize(
        "e, p, expected_g_per_kg",
        [
            (10.0, 1000.0, 6.28276),
            (23.37, 1000.0, 14.88382),
            (12.0, 1013.25, 7.45460),
            (1.0, 500.0, 1.24648),
        ],
    )
    def test_hand_checked_values(self, e, p, expected_g_per_kg):
        got = Vapor.get_mixing_ratio(e, p) * 1e3
        assert got == pytest.approx(expected_g_per_kg, abs=1e-5)

    def test_random_inputs_match_closed_form(self, random_inputs):
        e, p = random_inputs
        np.testing.assert_allclose(
            Vapor.get_mixing_ratio(e, p), ref_mixing_ratio(e, p), rtol=RTOL_EXACT
        )

    def test_inverse_recovers_vapor_pressure(self, random_inputs):
        e, p = random_inputs
        w = Vapor.get_mixing_ratio(e, p)
        np.testing.assert_allclose(ref_vapor_pressure(w, p), e, rtol=1e-11, atol=1e-12)


# ============================================================================
# 2. Physical properties
# ============================================================================


class TestPhysicalProperties:
    def test_zero_vapor_gives_zero(self):
        assert Vapor.get_mixing_ratio(0.0, 1000.0) == 0.0

    def test_non_negative_for_valid_input(self, random_inputs):
        assert np.all(Vapor.get_mixing_ratio(*random_inputs) >= 0.0)

    def test_strictly_increasing_in_vapor_pressure(self):
        e = np.linspace(0.0, 60.0, 500)
        w = Vapor.get_mixing_ratio(e, 1000.0)
        assert np.all(np.diff(w) > 0)

    def test_strictly_decreasing_in_total_pressure(self):
        p = np.linspace(100.0, 1050.0, 500)
        w = Vapor.get_mixing_ratio(10.0, p)
        assert np.all(np.diff(w) < 0)

    @pytest.mark.parametrize("k", [0.1, 0.5, 2.0, 10.0])
    def test_scale_invariant(self, k):
        # w depends only on e/p, so scaling both leaves it unchanged
        assert Vapor.get_mixing_ratio(k * 12.0, k * 1000.0) == pytest.approx(
            Vapor.get_mixing_ratio(12.0, 1000.0), rel=RTOL_EXACT
        )

    def test_small_e_limit_is_eps_e_over_p(self):
        e, p = 1e-6, 1000.0
        assert Vapor.get_mixing_ratio(e, p) == pytest.approx(eps * e / p, rel=1e-8)

    def test_exceeds_linear_approximation(self):
        # ε e/(p−e) > ε e/p for 0 < e < p
        e, p = 30.0, 1000.0
        assert Vapor.get_mixing_ratio(e, p) > eps * e / p

    def test_realistic_atmospheric_range(self, random_inputs):
        w = Vapor.get_mixing_ratio(*random_inputs)
        assert w.max() < 0.05  # < 50 g/kg for e/p ≤ 6 %


# ============================================================================
# 3. Unphysical input
# ============================================================================


class TestUnphysicalInput:
    def test_p_equal_e_returns_nan(self):
        assert math.isnan(Vapor.get_mixing_ratio(1000.0, 1000.0))

    def test_p_less_than_e_returns_nan(self):
        assert math.isnan(Vapor.get_mixing_ratio(50.0, 40.0))

    def test_nan_guard_is_element_wise(self):
        e = np.array([10.0, 1000.0, 50.0, 5.0])
        p = np.array([1000.0, 1000.0, 40.0, 500.0])
        got = Vapor.get_mixing_ratio(e, p)
        assert np.isfinite(got[[0, 3]]).all()
        assert np.isnan(got[[1, 2]]).all()
        assert got[0] == pytest.approx(ref_mixing_ratio(10.0, 1000.0), rel=RTOL_EXACT)

    def test_nan_input_propagates(self):
        got = Vapor.get_mixing_ratio(
            np.array([10.0, np.nan, 10.0]), np.array([1000.0, 1000.0, np.nan])
        )
        assert np.isfinite(got[0])
        assert np.isnan(got[1:]).all()

    def test_never_returns_negative(self):
        e = np.array([10.0, 999.9, 1000.0, 1500.0])
        got = Vapor.get_mixing_ratio(e, 1000.0)
        assert not np.any(got < 0)


# ============================================================================
# 4. Input handling
# ============================================================================


class TestInputHandling:
    @pytest.mark.parametrize(
        "e, p",
        [
            (12.0, 1013.25),
            (12, 1013),
            (np.float64(12.0), np.float64(1013.25)),
            (np.float32(12.0), 1013.25),
            (np.array(12.0), np.array(1013.25)),
        ],
        ids=["float", "int", "np_float64", "np_float32", "0d_array"],
    )
    def test_scalar_like_returns_python_float(self, e, p):
        got = Vapor.get_mixing_ratio(e, p)
        assert type(got) is float
        assert got == pytest.approx(ref_mixing_ratio(float(e), float(p)), rel=1e-6)

    def test_keyword_arguments(self):
        got = Vapor.get_mixing_ratio(vapor_pressure_hpa=12.0, pressure_hpa=1013.25)
        assert got == pytest.approx(ref_mixing_ratio(12.0, 1013.25), rel=RTOL_EXACT)

    @pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
    def test_sequence_input_returns_ndarray(self, container):
        got = Vapor.get_mixing_ratio(container(E_CASES), container(P_CASES))
        assert isinstance(got, np.ndarray)
        np.testing.assert_allclose(
            got, ref_mixing_ratio(E_CASES, P_CASES), rtol=RTOL_EXACT
        )

    def test_single_element_array_stays_array(self):
        got = Vapor.get_mixing_ratio(np.array([12.0]), np.array([1013.25]))
        assert isinstance(got, np.ndarray)
        assert got.shape == (1,)

    def test_empty_array(self):
        got = Vapor.get_mixing_ratio(np.array([]), np.array([]))
        assert isinstance(got, np.ndarray)
        assert got.shape == (0,)

    @pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 4)])
    def test_shape_and_dtype_preserved(self, shape):
        rng = np.random.default_rng(0)
        p = rng.uniform(300.0, 1050.0, shape)
        e = rng.uniform(0.0, 0.05, shape) * p
        got = Vapor.get_mixing_ratio(e, p)
        assert got.shape == shape
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, ref_mixing_ratio(e, p), rtol=RTOL_EXACT)

    def test_float32_input_returns_float64(self):
        got = Vapor.get_mixing_ratio(
            E_CASES.astype(np.float32), P_CASES.astype(np.float32)
        )
        assert got.dtype == np.float64

    def test_broadcast_array_e_scalar_p(self):
        got = Vapor.get_mixing_ratio(E_CASES, 850.0)
        assert got.shape == E_CASES.shape
        np.testing.assert_allclose(
            got, ref_mixing_ratio(E_CASES, 850.0), rtol=RTOL_EXACT
        )

    def test_broadcast_scalar_e_array_p(self):
        got = Vapor.get_mixing_ratio(5.0, P_CASES)
        assert got.shape == P_CASES.shape
        np.testing.assert_allclose(got, ref_mixing_ratio(5.0, P_CASES), rtol=RTOL_EXACT)

    def test_broadcast_pressure_levels_over_grid(self):
        # Typical reanalysis layout: e (level, lat, lon), p (level, 1, 1)
        rng = np.random.default_rng(3)
        p_levels = np.array([1000.0, 850.0, 500.0]).reshape(3, 1, 1)
        e_grid = rng.uniform(0.1, 20.0, (3, 4, 5))
        got = Vapor.get_mixing_ratio(e_grid, p_levels)
        assert got.shape == (3, 4, 5)
        np.testing.assert_allclose(
            got, ref_mixing_ratio(e_grid, p_levels), rtol=RTOL_EXACT
        )

    def test_broadcast_outer_product(self):
        e = np.array([[1.0], [10.0], [30.0]])  # (3, 1)
        p = np.array([500.0, 850.0, 1000.0])  # (3,)
        got = Vapor.get_mixing_ratio(e, p)
        assert got.shape == (3, 3)
        np.testing.assert_allclose(got, ref_mixing_ratio(e, p), rtol=RTOL_EXACT)

    @pytest.mark.parametrize(
        "e_shape, p_shape", [((3,), (4,)), ((2, 3), (3, 2)), ((5,), (2, 3))]
    )
    def test_incompatible_shapes_raise(self, e_shape, p_shape):
        with pytest.raises(ValueError):
            Vapor.get_mixing_ratio(np.ones(e_shape), np.full(p_shape, 1000.0))

    def test_non_contiguous_input(self):
        e = np.linspace(0.0, 30.0, 40)[::3]
        p = np.linspace(500.0, 1000.0, 40)[::3]
        assert not e.flags["C_CONTIGUOUS"]
        np.testing.assert_allclose(
            Vapor.get_mixing_ratio(e, p), ref_mixing_ratio(e, p), rtol=RTOL_EXACT
        )

    def test_fortran_ordered_input(self):
        e = np.asfortranarray(np.full((3, 4), 10.0))
        p = np.asfortranarray(np.linspace(500.0, 1000.0, 12).reshape(3, 4))
        np.testing.assert_allclose(
            Vapor.get_mixing_ratio(e, p), ref_mixing_ratio(e, p), rtol=RTOL_EXACT
        )

    def test_inputs_not_mutated(self):
        e, p = E_CASES.copy(), P_CASES.copy()
        Vapor.get_mixing_ratio(e, p)
        np.testing.assert_array_equal(e, E_CASES)
        np.testing.assert_array_equal(p, P_CASES)

    def test_array_matches_scalar_loop(self):
        vectorised = Vapor.get_mixing_ratio(E_CASES, P_CASES)
        looped = np.array(
            [
                Vapor.get_mixing_ratio(float(e), float(p))
                for e, p in zip(E_CASES, P_CASES)
            ]
        )
        np.testing.assert_array_equal(vectorised, looped)

    def test_deterministic(self, random_inputs):
        a = Vapor.get_mixing_ratio(*random_inputs)
        b = Vapor.get_mixing_ratio(*random_inputs)
        np.testing.assert_array_equal(a, b)

    def test_large_array(self):
        rng = np.random.default_rng(99)
        p = rng.uniform(100.0, 1050.0, 1_000_000)
        e = rng.uniform(0.0, 0.06, 1_000_000) * p
        got = Vapor.get_mixing_ratio(e, p)
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, ref_mixing_ratio(e, p), rtol=RTOL_EXACT)


# ============================================================================
# 5. Accuracy vs MetPy
# ============================================================================


@requires_metpy
class TestAgainstMetPy:
    @staticmethod
    def metpy_w(e, p):
        return (
            mp_mixing_ratio(np.asarray(e) * mp_units.hPa, np.asarray(p) * mp_units.hPa)
            .to("dimensionless")
            .magnitude
        )

    @pytest.mark.parametrize("label, e, p", MOIST_CASES, ids=CASE_IDS)
    def test_matches_metpy_mixing_ratio(self, label, e, p):
        ours = Vapor.get_mixing_ratio(e, p)
        ref = float(self.metpy_w(e, p))
        assert ours == pytest.approx(ref, rel=RTOL_EPS), (
            f"{label}: ours={ours * 1e3:.5f} g/kg metpy={ref * 1e3:.5f} g/kg"
        )

    def test_matches_metpy_on_random_inputs(self, random_inputs):
        e, p = random_inputs
        mask = e > 0  # relative error undefined at w = 0
        np.testing.assert_allclose(
            Vapor.get_mixing_ratio(e[mask], p[mask]),
            self.metpy_w(e[mask], p[mask]),
            rtol=RTOL_EPS,
        )

    def test_deviation_explained_by_eps(self):
        # Rescaling MetPy's result by our ε / MetPy's ε must give an exact match
        from metpy.constants import epsilon as mp_eps

        ratio = eps / float(mp_eps.magnitude)
        np.testing.assert_allclose(
            Vapor.get_mixing_ratio(E_CASES, P_CASES),
            self.metpy_w(E_CASES, P_CASES) * ratio,
            rtol=1e-10,
        )

    def test_round_trip_through_metpy_vapor_pressure(self):
        w = Vapor.get_mixing_ratio(E_CASES, P_CASES)
        e_back = (
            mp_vapor_pressure(P_CASES * mp_units.hPa, w * mp_units("kg/kg"))
            .to("hPa")
            .magnitude
        )
        np.testing.assert_allclose(e_back, E_CASES, rtol=RTOL_EPS)


# ============================================================================
# 6. Accuracy vs PsychroLib (ASHRAE Handbook 2017)
# ============================================================================


@requires_psychrolib
class TestAgainstPsychroLib:
    @pytest.mark.parametrize("label, e, p", MOIST_CASES, ids=CASE_IDS)
    def test_matches_ashrae_hum_ratio(self, label, e, p):
        ours = Vapor.get_mixing_ratio(e, p)
        ref = psychrolib.GetHumRatioFromVapPres(e * 100.0, p * 100.0)  # Pa
        # PsychroLib clamps to a minimum humidity ratio of 1e-7 kg/kg
        if ref <= 1e-7:
            pytest.skip("below PsychroLib minimum humidity ratio")
        assert ours == pytest.approx(ref, rel=RTOL_EPS), (
            f"{label}: ours={ours * 1e3:.5f} g/kg ashrae={ref * 1e3:.5f} g/kg"
        )

    @pytest.mark.parametrize("label, e, p", MOIST_CASES, ids=CASE_IDS)
    def test_round_trip_through_ashrae_vap_pres(self, label, e, p):
        w = Vapor.get_mixing_ratio(e, p)
        if w <= 1e-7:
            pytest.skip("below PsychroLib minimum humidity ratio")
        e_back = psychrolib.GetVapPresFromHumRatio(w, p * 100.0) / 100.0
        assert e_back == pytest.approx(e, rel=RTOL_EPS)


# ============================================================================
# 7. Saturation mixing ratio via get_vapor_saturation
# ============================================================================


class TestSaturationComposition:
    """ws(T, p) = get_mixing_ratio(get_vapor_saturation(T, water), p)."""

    @staticmethod
    def ours_ws(temp_c, p):
        es = Vapor.get_vapor_saturation(temp_c + 273.15, phase="water")
        return Vapor.get_mixing_ratio(es, p)

    def test_dewpoint_identity_ordering(self):
        # Actual w = ws(Td) must be below ws(T) whenever Td < T
        assert 0.0 < self.ours_ws(10.0, 1000.0) < self.ours_ws(20.0, 1000.0)

    def test_increases_with_temperature(self):
        temps = np.linspace(5.0, 35.0, 13)
        values = [self.ours_ws(t, 1000.0) for t in temps]
        assert np.all(np.diff(values) > 0)

    @requires_metpy
    @pytest.mark.parametrize("label, temp_c, p", SATURATION_CASES, ids=SAT_IDS)
    def test_matches_metpy_saturation_mixing_ratio(self, label, temp_c, p):
        ours = self.ours_ws(temp_c, p)
        ref = float(
            mp_saturation_mixing_ratio(p * mp_units.hPa, (temp_c + 273.15) * mp_units.K)
            .to("dimensionless")
            .magnitude
        )
        assert ours == pytest.approx(ref, rel=RTOL_SATURATION), (
            f"{label}: ours={ours * 1e3:.4f} g/kg metpy={ref * 1e3:.4f} g/kg"
        )

    @requires_psychrolib
    @pytest.mark.parametrize(
        "label, temp_c, p",
        [c for c in SATURATION_CASES if c[1] > 0.0],
        ids=[c[0] for c in SATURATION_CASES if c[1] > 0.0],
    )
    def test_matches_ashrae_sat_hum_ratio(self, label, temp_c, p):
        # ASHRAE switches to ice below 0.01 °C, so only liquid-water cases
        ours = self.ours_ws(temp_c, p)
        ref = psychrolib.GetSatHumRatio(temp_c, p * 100.0)
        assert ours == pytest.approx(ref, rel=RTOL_SATURATION), (
            f"{label}: ours={ours * 1e3:.4f} g/kg ashrae={ref * 1e3:.4f} g/kg"
        )
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])