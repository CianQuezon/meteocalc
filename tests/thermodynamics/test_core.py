"""
Tests for the public ``Thermodynamics`` API (``meteocalc.thermodynamics.core``)
===============================================================================

``Thermodynamics`` is the user-facing static namespace for potential
temperature. These tests treat it as a black box and verify it against:

* **Independent reference implementations** written here in plain Python
  (``math`` only), so a bug in the JIT kernels cannot hide behind itself.
* **MetPy** as an external reference implementation (skipped if absent).
* **Physical invariants** that must hold regardless of implementation.

Test sections
-------------
1.  get_modes_available      — contents, type, stability
2.  get_equation             — instance returned, mode configured, input forms,
                               invalid input, end-to-end calculation
3.  get_dry_potential_temperature        — reference values, physics, p0
4.  get_lcl_potential_temperature        — reference values, physics
5.  get_equivalent_potential_temperature — reference values, physics, nan guard
6.  get_potential_temperature dispatcher — routing, defaults, required inputs
7.  Input handling           — scalar/array types, shapes, broadcasting
8.  Accuracy vs MetPy        — dry θ and θE, isolated formula + full pipeline
"""

import math

import numpy as np
import pytest

from meteocalc.lcl.core import Lcl
from meteocalc.shared.constants import eps
from meteocalc.thermodynamics._enums import PotentialTemperatureMode
from meteocalc.thermodynamics._thermodynamic_equation import (
    BoltonPotentialTempEquation,
    PoissonEquation,
    PotentialTemperatureEquation,
)
from meteocalc.thermodynamics.core import Thermodynamics
from meteocalc.vapor.core import Vapor

try:
    from metpy.calc import (
        equivalent_potential_temperature as mp_theta_e,
    )
    from metpy.calc import (
        potential_temperature as mp_theta,
    )
    from metpy.calc import (
        saturation_mixing_ratio as mp_saturation_mixing_ratio,
    )
    from metpy.calc import (
        saturation_vapor_pressure as mp_saturation_vapor_pressure,
    )
    from metpy.units import units as mp_units

    HAS_METPY = True
except ImportError:
    HAS_METPY = False

requires_metpy = pytest.mark.skipif(not HAS_METPY, reason="metpy not installed")


# ============================================================================
# Constants and tolerances
# ============================================================================

KAPPA = 287.04 / 1004.0
"""Rd/cp used by the JIT kernels. Reference implementations below use it too."""

TOL_REFERENCE = 1e-9
"""Relative tolerance vs the in-file reference formulas (float64 round-off)."""

TOL_DRY_VS_METPY_K = 0.1
"""Dry θ vs MetPy. Only kappa differs (0.28590 vs 0.28571); measured max 0.04 K."""

RTOL_DRY_VS_METPY = 5e-4
"""Relative dry θ tolerance for deep profiles. The kappa mismatch scales with
ln(p0/p): ≈0.044 % (0.17 K) at 100 hPa, so a fixed-K tolerance is wrong aloft."""

TOL_THETA_E_FORMULA_VS_METPY_K = 0.1
"""θE vs MetPy with identical e, r and T_L inputs. Only kappa differs."""

TOL_THETA_E_PIPELINE_VS_METPY_K = 1.5
"""θE built from meteocalc's own Vapor (Goff-Gratch) + Lcl solver vs MetPy
(Ambaum es + Bolton eq. 15 LCL). Measured max deviation 0.73 K at 700 hPa."""

ALL_MODES = ["dry", "lcl", "equivalent"]
MOIST_MODES = ["lcl", "equivalent"]

ATMOSPHERIC_PROFILES = [
    # (label, temp_c, dewpoint_c, pressure_hpa)
    ("surface_warm_moist", 25.0, 18.0, 1013.25),
    ("surface_cool_dry", 10.0, 2.0, 1000.0),
    ("level_850hpa", 5.0, -5.0, 850.0),
    ("level_700hpa", -5.0, -15.0, 700.0),
    ("level_500hpa", -20.0, -30.0, 500.0),
    ("cold_dry_surface", -5.0, -15.0, 1013.25),
    ("hot_humid_tropical", 35.0, 28.0, 1010.0),
]
PROFILE_IDS = [p[0] for p in ATMOSPHERIC_PROFILES]


# ============================================================================
# Independent reference implementations (pure Python, no JIT)
# ============================================================================


def ref_dry_theta(temp_k: float, pressure_hpa: float, p0: float = 1000.0) -> float:
    """Poisson's equation θ = T (p0/p)^κ."""
    return temp_k * (p0 / pressure_hpa) ** KAPPA


def ref_theta_l(
    temp_k: float,
    lcl_temp_k: float,
    pressure_hpa: float,
    vapor_pressure_hpa: float,
    mixing_ratio: float,
    p0: float = 1000.0,
) -> float:
    """Bolton (1980) eq. 24: θL = T (p0/(p−e))^κ (T/T_L)^(0.28 r)."""
    return (
        temp_k
        * (p0 / (pressure_hpa - vapor_pressure_hpa)) ** KAPPA
        * (temp_k / lcl_temp_k) ** (0.28 * mixing_ratio)
    )


def ref_theta_e(
    temp_k: float,
    lcl_temp_k: float,
    pressure_hpa: float,
    vapor_pressure_hpa: float,
    mixing_ratio: float,
    p0: float = 1000.0,
) -> float:
    """Bolton (1980) eq. 39: θE = θL exp[(3036/T_L − 1.78) r (1 + 0.448 r)]."""
    theta_l = ref_theta_l(
        temp_k, lcl_temp_k, pressure_hpa, vapor_pressure_hpa, mixing_ratio, p0
    )
    exponent = (
        (3036.0 / lcl_temp_k - 1.78) * mixing_ratio * (1.0 + 0.448 * mixing_ratio)
    )
    return theta_l * math.exp(exponent)


def pipeline_inputs(temp_c: float, dewpoint_c: float, pressure_hpa: float) -> dict:
    """Moist inputs built from meteocalc's own Vapor + Lcl modules."""
    temp_k = temp_c + 273.15
    dewpoint_k = dewpoint_c + 273.15
    e = float(Vapor.get_vapor_saturation(dewpoint_k, equation="goff_gratch"))
    lcl_temp_k, _ = Lcl.get_lcl(
        temp_k=temp_k, dewpoint_temp_k=dewpoint_k, pressure_hpa=pressure_hpa
    )
    return {
        "temp_k": temp_k,
        "lcl_temp_k": float(lcl_temp_k),
        "pressure_hpa": pressure_hpa,
        "vapor_pressure_hpa": e,
        "mixing_ratio": eps * e / (pressure_hpa - e),
    }


def metpy_consistent_inputs(
    temp_c: float, dewpoint_c: float, pressure_hpa: float
) -> dict:
    """Moist inputs built exactly as MetPy builds them inside θE.

    Feeding these to meteocalc isolates the θE formula itself: the only
    remaining difference from MetPy is the kappa constant.
    """
    temp_k = temp_c + 273.15
    dewpoint_k = dewpoint_c + 273.15
    p = pressure_hpa * mp_units.hPa
    td = dewpoint_k * mp_units.K
    e = float(mp_saturation_vapor_pressure(td).to("hPa").magnitude)
    r = float(mp_saturation_mixing_ratio(p, td).to("dimensionless").magnitude)
    t_l = 56.0 + 1.0 / (
        1.0 / (dewpoint_k - 56.0) + math.log(temp_k / dewpoint_k) / 800.0
    )
    return {
        "temp_k": temp_k,
        "lcl_temp_k": t_l,
        "pressure_hpa": pressure_hpa,
        "vapor_pressure_hpa": e,
        "mixing_ratio": r,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def moist_parcel() -> dict:
    """Fixed moist parcel with hand-chosen, physically consistent inputs."""
    return {
        "temp_k": 293.15,
        "lcl_temp_k": 283.15,
        "pressure_hpa": 1013.25,
        "vapor_pressure_hpa": 12.0,
        "mixing_ratio": 0.008,
    }


@pytest.fixture
def moist_arrays() -> dict:
    """Random but physically plausible moist inputs, 1000 points."""
    rng = np.random.default_rng(42)
    n = 1000
    temp_k = rng.uniform(250.0, 310.0, n)
    lcl_temp_k = temp_k - rng.uniform(0.0, 15.0, n)
    pressure_hpa = rng.uniform(500.0, 1050.0, n)
    vapor_pressure_hpa = rng.uniform(1.0, 30.0, n)
    mixing_ratio = eps * vapor_pressure_hpa / (pressure_hpa - vapor_pressure_hpa)
    return {
        "temp_k": temp_k,
        "lcl_temp_k": lcl_temp_k,
        "pressure_hpa": pressure_hpa,
        "vapor_pressure_hpa": vapor_pressure_hpa,
        "mixing_ratio": mixing_ratio,
    }


# ============================================================================
# 1. get_modes_available
# ============================================================================


class TestGetModesAvailable:
    def test_returns_list_of_str(self):
        modes = Thermodynamics.get_modes_available()
        assert isinstance(modes, list)
        assert all(isinstance(m, str) for m in modes)

    def test_contains_exactly_the_three_modes(self):
        assert sorted(Thermodynamics.get_modes_available()) == sorted(ALL_MODES)

    def test_matches_enum_members(self):
        expected = {m.value for m in PotentialTemperatureMode}
        assert set(Thermodynamics.get_modes_available()) == expected

    def test_no_duplicates(self):
        modes = Thermodynamics.get_modes_available()
        assert len(modes) == len(set(modes))

    def test_stable_across_calls(self):
        assert (
            Thermodynamics.get_modes_available() == Thermodynamics.get_modes_available()
        )

    def test_returned_list_is_not_shared_state(self):
        modes = Thermodynamics.get_modes_available()
        modes.append("bogus")
        assert "bogus" not in Thermodynamics.get_modes_available()

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_every_listed_mode_is_accepted_by_get_equation(self, mode):
        assert mode in Thermodynamics.get_modes_available()
        Thermodynamics.get_equation(mode)


# ============================================================================
# 2. get_equation
# ============================================================================


class TestGetEquation:
    @pytest.mark.parametrize(
        "mode, expected_cls",
        [
            ("dry", PoissonEquation),
            ("lcl", BoltonPotentialTempEquation),
            ("equivalent", BoltonPotentialTempEquation),
        ],
    )
    def test_returns_instance_of_registered_class(self, mode, expected_cls):
        eq = Thermodynamics.get_equation(mode)
        assert isinstance(eq, expected_cls)
        assert isinstance(eq, PotentialTemperatureEquation)

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_instance_configured_for_requested_mode(self, mode):
        eq = Thermodynamics.get_equation(mode)
        assert eq.mode == PotentialTemperatureMode(mode)

    @pytest.mark.parametrize("mode", list(PotentialTemperatureMode))
    def test_accepts_enum_input(self, mode):
        eq = Thermodynamics.get_equation(mode)
        assert eq.mode == mode

    @pytest.mark.parametrize("mode", ["DRY", "Lcl", "EQUIVALENT"])
    def test_string_input_is_case_insensitive(self, mode):
        eq = Thermodynamics.get_equation(mode)
        assert eq.mode == PotentialTemperatureMode(mode.lower())

    def test_returns_fresh_instance_each_call(self):
        a = Thermodynamics.get_equation("equivalent")
        b = Thermodynamics.get_equation("equivalent")
        assert a is not b

    @pytest.mark.parametrize("bad", ["moist", "", "theta_e", " dry "])
    def test_invalid_string_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="Invalid enum"):
            Thermodynamics.get_equation(bad)

    @pytest.mark.parametrize("bad", [None, 1, 3.5, ["dry"]])
    def test_non_string_raises_type_error(self, bad):
        with pytest.raises(TypeError):
            Thermodynamics.get_equation(bad)

    def test_dry_equation_calculates_reference_value(self):
        eq = Thermodynamics.get_equation("dry")
        got = eq.calculate(temp_k=288.15, pressure_hpa=850.0)
        assert got == pytest.approx(ref_dry_theta(288.15, 850.0), rel=TOL_REFERENCE)

    def test_lcl_equation_calculates_theta_l_not_theta_e(self, moist_parcel):
        eq = Thermodynamics.get_equation("lcl")
        got = eq.calculate(**moist_parcel)
        assert got == pytest.approx(ref_theta_l(**moist_parcel), rel=TOL_REFERENCE)
        assert got != pytest.approx(ref_theta_e(**moist_parcel), rel=1e-3)

    def test_equivalent_equation_calculates_theta_e(self, moist_parcel):
        eq = Thermodynamics.get_equation("equivalent")
        got = eq.calculate(**moist_parcel)
        assert got == pytest.approx(ref_theta_e(**moist_parcel), rel=TOL_REFERENCE)

    @pytest.mark.parametrize("mode", MOIST_MODES)
    def test_matches_dedicated_public_method(self, mode, moist_parcel):
        direct = {
            "lcl": Thermodynamics.get_lcl_potential_temperature,
            "equivalent": Thermodynamics.get_equivalent_potential_temperature,
        }[mode]
        eq = Thermodynamics.get_equation(mode)
        assert eq.calculate(**moist_parcel) == direct(**moist_parcel)


# ============================================================================
# 3. get_dry_potential_temperature
# ============================================================================


class TestDryPotentialTemperature:
    @pytest.mark.parametrize(
        "temp_k, pressure_hpa",
        [
            (288.15, 850.0),
            (300.0, 1013.25),
            (250.0, 500.0),
            (220.0, 250.0),
            (210.0, 100.0),
            (310.0, 1050.0),
        ],
    )
    def test_matches_reference_formula(self, temp_k, pressure_hpa):
        got = Thermodynamics.get_dry_potential_temperature(temp_k, pressure_hpa)
        assert got == pytest.approx(
            ref_dry_theta(temp_k, pressure_hpa), rel=TOL_REFERENCE
        )

    def test_known_value(self):
        # 288.15 K at 850 hPa — hand-checked value
        got = Thermodynamics.get_dry_potential_temperature(288.15, 850.0)
        assert got == pytest.approx(301.8543925, abs=1e-6)

    @pytest.mark.parametrize("temp_k", [200.0, 273.15, 288.15, 320.0])
    def test_equals_temperature_at_reference_pressure(self, temp_k):
        got = Thermodynamics.get_dry_potential_temperature(temp_k, 1000.0)
        assert got == pytest.approx(temp_k, rel=1e-15)

    def test_greater_than_temperature_above_reference_level(self):
        assert Thermodynamics.get_dry_potential_temperature(280.0, 700.0) > 280.0

    def test_less_than_temperature_below_reference_level(self):
        assert Thermodynamics.get_dry_potential_temperature(280.0, 1050.0) < 280.0

    def test_monotonic_decreasing_in_pressure(self):
        p = np.linspace(100.0, 1050.0, 200)
        theta = Thermodynamics.get_dry_potential_temperature(280.0, p)
        assert np.all(np.diff(theta) < 0)

    def test_linear_in_temperature(self):
        theta_1 = Thermodynamics.get_dry_potential_temperature(250.0, 700.0)
        theta_2 = Thermodynamics.get_dry_potential_temperature(500.0, 700.0)
        assert theta_2 == pytest.approx(2.0 * theta_1, rel=1e-14)

    @pytest.mark.parametrize("p0", [1000.0, 1013.25, 850.0])
    def test_custom_reference_pressure(self, p0):
        got = Thermodynamics.get_dry_potential_temperature(288.15, 700.0, p0=p0)
        assert got == pytest.approx(ref_dry_theta(288.15, 700.0, p0), rel=TOL_REFERENCE)

    def test_custom_p0_equals_temperature_at_that_level(self):
        got = Thermodynamics.get_dry_potential_temperature(288.15, 1013.25, p0=1013.25)
        assert got == pytest.approx(288.15, rel=1e-15)


# ============================================================================
# 4. get_lcl_potential_temperature
# ============================================================================


class TestLclPotentialTemperature:
    def test_matches_reference_formula(self, moist_parcel):
        got = Thermodynamics.get_lcl_potential_temperature(**moist_parcel)
        assert got == pytest.approx(ref_theta_l(**moist_parcel), rel=TOL_REFERENCE)

    def test_known_value(self, moist_parcel):
        got = Thermodynamics.get_lcl_potential_temperature(**moist_parcel)
        assert got == pytest.approx(293.0681047, abs=1e-6)

    @pytest.mark.parametrize(
        "label, temp_c, dewpoint_c, pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_matches_reference_across_profiles(
        self, label, temp_c, dewpoint_c, pressure_hpa
    ):
        inputs = pipeline_inputs(temp_c, dewpoint_c, pressure_hpa)
        got = Thermodynamics.get_lcl_potential_temperature(**inputs)
        assert got == pytest.approx(ref_theta_l(**inputs), rel=TOL_REFERENCE)

    def test_reduces_to_dry_theta_when_dry(self):
        # e = 0, r = 0  →  θL = T (p0/p)^κ
        got = Thermodynamics.get_lcl_potential_temperature(
            temp_k=290.0,
            lcl_temp_k=270.0,
            pressure_hpa=850.0,
            vapor_pressure_hpa=0.0,
            mixing_ratio=0.0,
        )
        assert got == pytest.approx(
            Thermodynamics.get_dry_potential_temperature(290.0, 850.0), rel=1e-14
        )

    def test_exceeds_dry_theta_for_moist_air(self, moist_parcel):
        theta_l = Thermodynamics.get_lcl_potential_temperature(**moist_parcel)
        theta = Thermodynamics.get_dry_potential_temperature(
            moist_parcel["temp_k"], moist_parcel["pressure_hpa"]
        )
        assert theta_l > theta

    def test_custom_reference_pressure(self, moist_parcel):
        got = Thermodynamics.get_lcl_potential_temperature(**moist_parcel, p0=1013.25)
        assert got == pytest.approx(
            ref_theta_l(**moist_parcel, p0=1013.25), rel=TOL_REFERENCE
        )


# ============================================================================
# 5. get_equivalent_potential_temperature
# ============================================================================


class TestEquivalentPotentialTemperature:
    def test_matches_reference_formula(self, moist_parcel):
        got = Thermodynamics.get_equivalent_potential_temperature(**moist_parcel)
        assert got == pytest.approx(ref_theta_e(**moist_parcel), rel=TOL_REFERENCE)

    def test_known_value(self, moist_parcel):
        got = Thermodynamics.get_equivalent_potential_temperature(**moist_parcel)
        assert got == pytest.approx(314.8824105, abs=1e-6)

    @pytest.mark.parametrize(
        "label, temp_c, dewpoint_c, pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_matches_reference_across_profiles(
        self, label, temp_c, dewpoint_c, pressure_hpa
    ):
        inputs = pipeline_inputs(temp_c, dewpoint_c, pressure_hpa)
        got = Thermodynamics.get_equivalent_potential_temperature(**inputs)
        assert got == pytest.approx(ref_theta_e(**inputs), rel=TOL_REFERENCE)

    def test_equals_theta_l_when_dry(self):
        dry = {
            "temp_k": 290.0,
            "lcl_temp_k": 270.0,
            "pressure_hpa": 850.0,
            "vapor_pressure_hpa": 0.0,
            "mixing_ratio": 0.0,
        }
        assert Thermodynamics.get_equivalent_potential_temperature(
            **dry
        ) == pytest.approx(
            Thermodynamics.get_lcl_potential_temperature(**dry), rel=1e-14
        )

    @pytest.mark.parametrize(
        "label, temp_c, dewpoint_c, pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_ordering_theta_le_theta_l_le_theta_e(
        self, label, temp_c, dewpoint_c, pressure_hpa
    ):
        inputs = pipeline_inputs(temp_c, dewpoint_c, pressure_hpa)
        theta = Thermodynamics.get_dry_potential_temperature(
            inputs["temp_k"], pressure_hpa
        )
        theta_l = Thermodynamics.get_lcl_potential_temperature(**inputs)
        theta_e = Thermodynamics.get_equivalent_potential_temperature(**inputs)
        assert theta <= theta_l <= theta_e

    def test_increases_with_moisture(self):
        dewpoints_c = np.linspace(-20.0, 24.0, 12)
        values = [
            Thermodynamics.get_equivalent_potential_temperature(
                **pipeline_inputs(25.0, td, 1000.0)
            )
            for td in dewpoints_c
        ]
        assert np.all(np.diff(values) > 0)

    def test_returns_nan_beyond_exponent_guard(self):
        # (3036/60 − 1.78) · 1.5 · (1 + 0.448·1.5) ≈ 122 > 50 → nan
        got = Thermodynamics.get_equivalent_potential_temperature(
            temp_k=300.0,
            lcl_temp_k=60.0,
            pressure_hpa=1000.0,
            vapor_pressure_hpa=10.0,
            mixing_ratio=1.5,
        )
        assert math.isnan(got)

    def test_nan_guard_is_elementwise(self, moist_parcel):
        inputs = {k: np.array([v, v]) for k, v in moist_parcel.items()}
        inputs["lcl_temp_k"] = np.array([283.15, 60.0])
        inputs["mixing_ratio"] = np.array([0.008, 1.5])
        got = Thermodynamics.get_equivalent_potential_temperature(**inputs)
        assert np.isfinite(got[0])
        assert np.isnan(got[1])

    def test_custom_reference_pressure(self, moist_parcel):
        got = Thermodynamics.get_equivalent_potential_temperature(
            **moist_parcel, p0=1013.25
        )
        assert got == pytest.approx(
            ref_theta_e(**moist_parcel, p0=1013.25), rel=TOL_REFERENCE
        )


# ============================================================================
# 6. get_potential_temperature dispatcher
# ============================================================================


class TestDispatcher:
    def test_default_mode_is_dry(self):
        assert Thermodynamics.get_potential_temperature(
            288.15, 850.0
        ) == Thermodynamics.get_dry_potential_temperature(288.15, 850.0)

    @pytest.mark.parametrize("mode", ["dry", PotentialTemperatureMode.DRY, "DRY"])
    def test_dry_routes_to_dry_method(self, mode):
        got = Thermodynamics.get_potential_temperature(288.15, 850.0, mode=mode)
        assert got == Thermodynamics.get_dry_potential_temperature(288.15, 850.0)

    @pytest.mark.parametrize("mode", ["lcl", PotentialTemperatureMode.LCL, "LCL"])
    def test_lcl_routes_to_lcl_method(self, mode, moist_parcel):
        got = Thermodynamics.get_potential_temperature(mode=mode, **moist_parcel)
        assert got == Thermodynamics.get_lcl_potential_temperature(**moist_parcel)

    @pytest.mark.parametrize(
        "mode", ["equivalent", PotentialTemperatureMode.EQUIVALENT, "Equivalent"]
    )
    def test_equivalent_routes_to_equivalent_method(self, mode, moist_parcel):
        got = Thermodynamics.get_potential_temperature(mode=mode, **moist_parcel)
        assert got == Thermodynamics.get_equivalent_potential_temperature(
            **moist_parcel
        )

    def test_moist_modes_give_different_results(self, moist_parcel):
        theta_l = Thermodynamics.get_potential_temperature(mode="lcl", **moist_parcel)
        theta_e = Thermodynamics.get_potential_temperature(
            mode="equivalent", **moist_parcel
        )
        assert theta_l != pytest.approx(theta_e, rel=1e-3)

    def test_dry_mode_ignores_moisture_arguments(self, moist_parcel):
        got = Thermodynamics.get_potential_temperature(mode="dry", **moist_parcel)
        assert got == Thermodynamics.get_dry_potential_temperature(
            moist_parcel["temp_k"], moist_parcel["pressure_hpa"]
        )

    def test_p0_forwarded_to_dry(self):
        got = Thermodynamics.get_potential_temperature(288.15, 700.0, p0=1013.25)
        assert got == pytest.approx(
            ref_dry_theta(288.15, 700.0, 1013.25), rel=TOL_REFERENCE
        )

    @pytest.mark.parametrize("mode", MOIST_MODES)
    def test_p0_forwarded_to_moist(self, mode, moist_parcel):
        ref = {"lcl": ref_theta_l, "equivalent": ref_theta_e}[mode]
        got = Thermodynamics.get_potential_temperature(
            mode=mode, p0=1013.25, **moist_parcel
        )
        assert got == pytest.approx(ref(**moist_parcel, p0=1013.25), rel=TOL_REFERENCE)

    @pytest.mark.parametrize("mode", MOIST_MODES)
    @pytest.mark.parametrize(
        "missing", ["lcl_temp_k", "vapor_pressure_hpa", "mixing_ratio"]
    )
    def test_missing_single_moist_input_raises(self, mode, missing, moist_parcel):
        kwargs = {k: v for k, v in moist_parcel.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            Thermodynamics.get_potential_temperature(mode=mode, **kwargs)

    @pytest.mark.parametrize("mode", MOIST_MODES)
    def test_missing_all_moist_inputs_lists_every_name(self, mode):
        with pytest.raises(ValueError) as exc_info:
            Thermodynamics.get_potential_temperature(293.15, 1000.0, mode=mode)
        message = str(exc_info.value)
        assert mode in message
        for name in ("lcl_temp_k", "vapor_pressure_hpa", "mixing_ratio"):
            assert name in message

    def test_explicit_none_counts_as_missing(self, moist_parcel):
        kwargs = dict(moist_parcel, mixing_ratio=None)
        with pytest.raises(ValueError, match="mixing_ratio"):
            Thermodynamics.get_potential_temperature(mode="equivalent", **kwargs)

    def test_zero_moisture_is_not_treated_as_missing(self):
        got = Thermodynamics.get_potential_temperature(
            290.0,
            850.0,
            mode="equivalent",
            lcl_temp_k=270.0,
            vapor_pressure_hpa=0.0,
            mixing_ratio=0.0,
        )
        assert math.isfinite(got)

    @pytest.mark.parametrize("bad", ["moist", "theta", ""])
    def test_invalid_mode_string_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="Invalid enum"):
            Thermodynamics.get_potential_temperature(288.15, 850.0, mode=bad)

    @pytest.mark.parametrize("bad", [None, 0, 1.0])
    def test_invalid_mode_type_raises_type_error(self, bad):
        with pytest.raises(TypeError):
            Thermodynamics.get_potential_temperature(288.15, 850.0, mode=bad)


# ============================================================================
# 7. Input handling — types, shapes, broadcasting
# ============================================================================


class TestInputHandling:
    @pytest.mark.parametrize(
        "temp_k",
        [288.15, 288, np.float64(288.15), np.float32(288.15), np.array(288.15)],
    )
    def test_scalar_like_input_returns_python_float(self, temp_k):
        got = Thermodynamics.get_dry_potential_temperature(temp_k, 850.0)
        assert type(got) is float

    def test_moist_scalar_returns_python_float(self, moist_parcel):
        assert (
            type(Thermodynamics.get_lcl_potential_temperature(**moist_parcel)) is float
        )
        assert (
            type(Thermodynamics.get_equivalent_potential_temperature(**moist_parcel))
            is float
        )

    def test_integer_input_matches_float_input(self):
        assert Thermodynamics.get_dry_potential_temperature(
            288, 850
        ) == Thermodynamics.get_dry_potential_temperature(288.0, 850.0)

    def test_list_input_returns_ndarray(self):
        got = Thermodynamics.get_dry_potential_temperature([280.0, 290.0], 850.0)
        assert isinstance(got, np.ndarray)
        assert got.shape == (2,)

    def test_single_element_array_stays_array(self):
        got = Thermodynamics.get_dry_potential_temperature(np.array([288.15]), 850.0)
        assert isinstance(got, np.ndarray)
        assert got.shape == (1,)

    @pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
    def test_shape_preserved(self, shape):
        temps = np.full(shape, 288.15)
        got = Thermodynamics.get_dry_potential_temperature(temps, 850.0)
        assert got.shape == shape
        assert got.dtype == np.float64

    def test_scalar_and_array_broadcast(self):
        p = np.array([1000.0, 850.0, 700.0, 500.0])
        got = Thermodynamics.get_dry_potential_temperature(288.15, p)
        expected = np.array([ref_dry_theta(288.15, pi) for pi in p])
        np.testing.assert_allclose(got, expected, rtol=TOL_REFERENCE)

    def test_outer_product_broadcast(self):
        temps = np.array([[270.0], [290.0], [310.0]])  # (3, 1)
        p = np.array([1000.0, 850.0, 500.0])  # (3,)
        got = Thermodynamics.get_dry_potential_temperature(temps, p)
        assert got.shape == (3, 3)
        np.testing.assert_allclose(
            got[2, 1], ref_dry_theta(310.0, 850.0), rtol=TOL_REFERENCE
        )

    def test_incompatible_shapes_raise(self):
        with pytest.raises(ValueError):
            Thermodynamics.get_dry_potential_temperature(
                np.ones(3) * 288.0, np.ones(4) * 850.0
            )

    def test_input_array_not_mutated(self):
        temps = np.array([280.0, 290.0, 300.0])
        original = temps.copy()
        Thermodynamics.get_dry_potential_temperature(temps, 850.0)
        np.testing.assert_array_equal(temps, original)

    def test_non_contiguous_input(self):
        temps = np.linspace(250.0, 310.0, 20)[::2]
        assert not temps.flags["C_CONTIGUOUS"]
        got = Thermodynamics.get_dry_potential_temperature(temps, 850.0)
        expected = np.array([ref_dry_theta(t, 850.0) for t in temps])
        np.testing.assert_allclose(got, expected, rtol=TOL_REFERENCE)

    @pytest.mark.parametrize(
        "method",
        [
            Thermodynamics.get_lcl_potential_temperature,
            Thermodynamics.get_equivalent_potential_temperature,
        ],
        ids=["lcl", "equivalent"],
    )
    def test_moist_array_matches_scalar_loop(self, method, moist_arrays):
        vectorised = method(**moist_arrays)
        n = len(moist_arrays["temp_k"])
        looped = np.array(
            [
                method(**{k: float(v[i]) for k, v in moist_arrays.items()})
                for i in range(n)
            ]
        )
        np.testing.assert_allclose(vectorised, looped, rtol=1e-14)

    def test_moist_array_all_finite(self, moist_arrays):
        got = Thermodynamics.get_equivalent_potential_temperature(**moist_arrays)
        assert got.shape == moist_arrays["temp_k"].shape
        assert np.all(np.isfinite(got))

    def test_moist_partial_broadcast(self, moist_parcel):
        temps = np.array([290.0, 293.15, 296.0])
        inputs = dict(moist_parcel, temp_k=temps)
        got = Thermodynamics.get_equivalent_potential_temperature(**inputs)
        expected = np.array(
            [ref_theta_e(**dict(moist_parcel, temp_k=t)) for t in temps]
        )
        np.testing.assert_allclose(got, expected, rtol=TOL_REFERENCE)

    def test_nan_input_propagates(self):
        got = Thermodynamics.get_dry_potential_temperature(
            np.array([288.15, np.nan]), 850.0
        )
        assert np.isfinite(got[0])
        assert np.isnan(got[1])

    def test_large_array(self):
        rng = np.random.default_rng(7)
        temps = rng.uniform(200.0, 320.0, 200_000)
        p = rng.uniform(100.0, 1050.0, 200_000)
        got = Thermodynamics.get_dry_potential_temperature(temps, p)
        np.testing.assert_allclose(got, temps * (1000.0 / p) ** KAPPA, rtol=1e-12)


# ============================================================================
# 8. Accuracy vs MetPy
# ============================================================================


@requires_metpy
class TestAccuracyAgainstMetPy:
    @pytest.mark.parametrize(
        "label, temp_c, dewpoint_c, pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_dry_theta(self, label, temp_c, dewpoint_c, pressure_hpa):
        temp_k = temp_c + 273.15
        ours = Thermodynamics.get_dry_potential_temperature(temp_k, pressure_hpa)
        ref = float(
            mp_theta(pressure_hpa * mp_units.hPa, temp_k * mp_units.K).to("K").magnitude
        )
        assert abs(ours - ref) < TOL_DRY_VS_METPY_K, (
            f"{label}: ours={ours:.4f} K metpy={ref:.4f} K"
        )

    @pytest.mark.parametrize(
        "label, temp_c, dewpoint_c, pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_theta_e_formula_with_identical_inputs(
        self, label, temp_c, dewpoint_c, pressure_hpa
    ):
        inputs = metpy_consistent_inputs(temp_c, dewpoint_c, pressure_hpa)
        ours = Thermodynamics.get_equivalent_potential_temperature(**inputs)
        ref = float(
            mp_theta_e(
                pressure_hpa * mp_units.hPa,
                (temp_c + 273.15) * mp_units.K,
                (dewpoint_c + 273.15) * mp_units.K,
            )
            .to("K")
            .magnitude
        )
        assert abs(ours - ref) < TOL_THETA_E_FORMULA_VS_METPY_K, (
            f"{label}: ours={ours:.4f} K metpy={ref:.4f} K"
        )

    @pytest.mark.parametrize(
        "label, temp_c, dewpoint_c, pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_theta_e_full_pipeline(self, label, temp_c, dewpoint_c, pressure_hpa):
        inputs = pipeline_inputs(temp_c, dewpoint_c, pressure_hpa)
        ours = Thermodynamics.get_equivalent_potential_temperature(**inputs)
        ref = float(
            mp_theta_e(
                pressure_hpa * mp_units.hPa,
                (temp_c + 273.15) * mp_units.K,
                (dewpoint_c + 273.15) * mp_units.K,
            )
            .to("K")
            .magnitude
        )
        assert abs(ours - ref) < TOL_THETA_E_PIPELINE_VS_METPY_K, (
            f"{label}: ours={ours:.4f} K metpy={ref:.4f} K"
        )

    def test_dry_theta_vectorised_profile(self):
        p = np.linspace(100.0, 1050.0, 50)
        t = np.linspace(210.0, 300.0, 50)
        ours = Thermodynamics.get_dry_potential_temperature(t, p)
        ref = mp_theta(p * mp_units.hPa, t * mp_units.K).to("K").magnitude
        np.testing.assert_allclose(ours, ref, rtol=RTOL_DRY_VS_METPY, atol=0.0)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
