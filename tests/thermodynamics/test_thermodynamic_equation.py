"""
Test suite for meteocalc.thermodynamics._thermodynamic_equation.

Covers the equation *class* layer (broadcasting, scalar/array dispatch, mode
selection, shape preservation) rather than the underlying JIT kernels, which
are tested separately in ``test_poisson_scalar.py`` and
``test_poisson_vectorised.py``.

Accuracy is cross-validated against two independent professional references:

- **MetPy** (UCAR/Unidata) — dry potential temperature and equivalent
  potential temperature. MetPy implements Bolton (1980) eq. 43, which
  Davies-Jones (2009) identifies as the most accurate non-iterative
  theta-e formulation.
- **PsychroLib** (ASHRAE) — saturation vapour pressure and mixing ratio,
  the inputs that feed theta-L and theta-E.

Units follow the library convention: temperature in Kelvin, pressure in hPa,
mixing ratio in kg/kg. PsychroLib uses degC and Pa, so conversions are
applied at the comparison boundary.
"""

import numpy as np
import pytest

from meteocalc.lcl.core import Lcl
from meteocalc.shared.constants import eps
from meteocalc.thermodynamics._enums import (
    PotentialTemperatureMode,
    ThermodynamicEquationName,
)
from meteocalc.thermodynamics._thermodynamic_equation import (
    BoltonPotentialTempEquation,
    PoissonEquation,
    PotentialTemperatureEquation,
)
from meteocalc.vapor.core import Vapor

try:
    from metpy.calc import equivalent_potential_temperature, potential_temperature
    from metpy.units import units as metpy_units

    HAS_METPY = True
except ImportError:
    HAS_METPY = False

try:
    import psychrolib

    psychrolib.SetUnitSystem(psychrolib.SI)
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False


# ============================================================================
# Tolerances
# ============================================================================

TOL_THETA_K = 0.2
"""Dry theta vs MetPy. Measured max deviation 0.04 K (kappa constant differs)."""

TOL_THETA_E_K = 1.5
"""Theta-E vs MetPy. Measured max deviation 0.73 K (LCL solver vs closed form)."""

TOL_VAPOR_PA = 10.0
"""Vapour pressure vs PsychroLib. Measured max deviation 4.8 Pa (Goff-Gratch vs ASHRAE)."""

TOL_MIXING_RATIO = 1e-4
"""Mixing ratio vs PsychroLib. Measured max deviation 3e-5 kg/kg."""

TOL_EXACT = 1e-12
"""Scalar and vectorised JIT paths must agree to machine precision."""


# ============================================================================
# Shared test data
# ============================================================================

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


def actual_vapor_pressure_hpa(dewpoint_k, equation="goff_gratch"):
    """Actual vapour pressure e = es(Td), in hPa."""
    return float(Vapor.get_vapor_saturation(dewpoint_k, equation=equation))


def mixing_ratio_kgkg(dewpoint_k, pressure_hpa, equation="goff_gratch"):
    """Surface mixing ratio w = eps * e / (p - e), in kg/kg."""
    e = actual_vapor_pressure_hpa(dewpoint_k, equation=equation)
    return eps * e / (pressure_hpa - e)


def moist_parcel_inputs(temp_c, dewpoint_c, pressure_hpa):
    """
    Build the full five-argument input set for a moist potential temperature.

    Returns
    -------
    dict
        Keys: temp_k, lcl_temp_k, pressure_hpa, vapor_pressure_hpa, mixing_ratio.
    """
    temp_k = temp_c + 273.15
    dewpoint_k = dewpoint_c + 273.15
    lcl_temp_k, _ = Lcl.get_lcl(
        temp_k=temp_k, dewpoint_temp_k=dewpoint_k, pressure_hpa=pressure_hpa
    )
    return {
        "temp_k": temp_k,
        "lcl_temp_k": lcl_temp_k,
        "pressure_hpa": pressure_hpa,
        "vapor_pressure_hpa": actual_vapor_pressure_hpa(dewpoint_k),
        "mixing_ratio": mixing_ratio_kgkg(dewpoint_k, pressure_hpa),
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def poisson():
    """Dry potential temperature equation instance."""
    return PoissonEquation()


@pytest.fixture
def theta_l():
    """Bolton theta-L (potential temperature at LCL) instance."""
    return BoltonPotentialTempEquation(mode=PotentialTemperatureMode.LCL)


@pytest.fixture
def theta_e():
    """Bolton theta-E (equivalent potential temperature) instance."""
    return BoltonPotentialTempEquation(mode=PotentialTemperatureMode.EQUIVALENT)


@pytest.fixture
def standard_parcel():
    """Warm moist surface parcel: 25 degC, Td 18 degC, 1013.25 hPa."""
    return moist_parcel_inputs(25.0, 18.0, 1013.25)


# ============================================================================
# PoissonEquation — API contract and dispatch
# ============================================================================


@pytest.mark.unit
class TestPoissonEquationContract:
    """Class attributes and inheritance for PoissonEquation."""

    def test_is_potential_temperature_equation(self, poisson):
        assert isinstance(poisson, PotentialTemperatureEquation)

    def test_class_attributes(self, poisson):
        assert poisson.name is ThermodynamicEquationName.POTENTIAL_TEMP
        assert poisson.mode is PotentialTemperatureMode.DRY


@pytest.mark.unit
class TestPoissonDispatch:
    """Scalar/array dispatch, broadcasting, and shape preservation."""

    def test_scalar_returns_python_float(self, poisson):
        result = poisson.calculate(temp_k=288.15, pressure_hpa=850.0)

        assert isinstance(result, float), f"expected float, got {type(result)}"
        assert np.isfinite(result)

    def test_array_returns_ndarray_same_shape(self, poisson):
        temps = np.array([273.15, 283.15, 293.15])
        pressures = np.array([1000.0, 850.0, 700.0])

        result = poisson.calculate(temp_k=temps, pressure_hpa=pressures)

        assert isinstance(result, np.ndarray)
        assert result.shape == temps.shape
        assert result.dtype == np.float64

    @pytest.mark.parametrize("shape", [(2, 3), (4, 2), (2, 2, 2)])
    def test_multidimensional_shape_preserved(self, poisson, shape):
        """Flatten/reshape round trip must restore the original shape."""
        temps = np.full(shape, 288.15)
        pressures = np.full(shape, 850.0)

        result = poisson.calculate(temp_k=temps, pressure_hpa=pressures)

        assert result.shape == shape

    def test_scalar_array_broadcast(self, poisson):
        """A scalar temperature broadcasts against an array of pressures."""
        pressures = np.array([1000.0, 850.0, 500.0])

        result = poisson.calculate(temp_k=288.15, pressure_hpa=pressures)

        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_scalar_and_vector_paths_agree(self, poisson):
        """
        The scalar JIT kernel and the vectorised kernel must produce
        identical results for identical inputs.
        """
        temps = np.array([273.15, 288.15, 303.15])
        pressures = np.array([1000.0, 850.0, 700.0])

        vector_result = poisson.calculate(temp_k=temps, pressure_hpa=pressures)
        scalar_result = np.array(
            [
                poisson.calculate(temp_k=float(t), pressure_hpa=float(p))
                for t, p in zip(temps, pressures)
            ]
        )

        np.testing.assert_allclose(vector_result, scalar_result, rtol=0, atol=TOL_EXACT)

    def test_custom_reference_pressure(self, poisson):
        """p0 is a calculation convention, not per-element data."""
        result_1000 = poisson.calculate(temp_k=288.15, pressure_hpa=850.0, p0=1000.0)
        result_500 = poisson.calculate(temp_k=288.15, pressure_hpa=850.0, p0=500.0)

        assert result_1000 != result_500
        assert result_500 < result_1000, "lower p0 must lower theta"


@pytest.mark.unit
class TestPoissonPhysics:
    """Analytical properties that hold regardless of implementation."""

    def test_theta_equals_temperature_at_reference_pressure(self, poisson):
        """At p == p0 the Poisson exponent vanishes, so theta == T."""
        result = poisson.calculate(temp_k=288.15, pressure_hpa=1000.0, p0=1000.0)

        assert result == pytest.approx(288.15, abs=1e-9)

    def test_theta_increases_as_pressure_decreases(self, poisson):
        """Monotonic in pressure at fixed temperature."""
        pressures = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])

        thetas = poisson.calculate(temp_k=288.15, pressure_hpa=pressures)

        assert np.all(np.diff(thetas) > 0), f"not monotonic: {thetas}"

    def test_theta_increases_with_temperature(self, poisson):
        """Monotonic in temperature at fixed pressure."""
        temps = np.array([253.15, 273.15, 293.15, 313.15])

        thetas = poisson.calculate(temp_k=temps, pressure_hpa=850.0)

        assert np.all(np.diff(thetas) > 0), f"not monotonic: {thetas}"

    @pytest.mark.parametrize(
        "label,temp_c,dewpoint_c,pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_theta_positive_and_realistic(
        self, poisson, label, temp_c, dewpoint_c, pressure_hpa
    ):
        result = poisson.calculate(temp_k=temp_c + 273.15, pressure_hpa=pressure_hpa)

        assert 150.0 < result < 500.0, f"{label}: unrealistic theta {result} K"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestPoissonAgainstMetPy:
    """Cross-validation of dry theta against MetPy (UCAR/Unidata)."""

    @pytest.mark.parametrize(
        "label,temp_c,dewpoint_c,pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_matches_metpy_scalar(
        self, poisson, label, temp_c, dewpoint_c, pressure_hpa
    ):
        temp_k = temp_c + 273.15

        ours = poisson.calculate(temp_k=temp_k, pressure_hpa=pressure_hpa)
        reference = float(
            potential_temperature(
                pressure_hpa * metpy_units.hPa, temp_k * metpy_units.K
            )
            .to("K")
            .magnitude
        )

        assert ours == pytest.approx(reference, abs=TOL_THETA_K), (
            f"{label}: ours={ours:.4f} K metpy={reference:.4f} K "
            f"err={abs(ours - reference):.4f} K"
        )

    def test_matches_metpy_vertical_profile(self, poisson):
        """Full sounding from surface to tropopause, array path."""
        pressures = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0])
        temps_k = np.array([15.0, 10.0, 5.0, -5.0, -20.0, -45.0, -60.0]) + 273.15

        ours = poisson.calculate(temp_k=temps_k, pressure_hpa=pressures)
        reference = np.array(
            [
                float(
                    potential_temperature(p * metpy_units.hPa, t * metpy_units.K)
                    .to("K")
                    .magnitude
                )
                for t, p in zip(temps_k, pressures)
            ]
        )

        np.testing.assert_allclose(ours, reference, atol=TOL_THETA_K)


# ============================================================================
# BoltonPotentialTempEquation — construction and mode selection
# ============================================================================


@pytest.mark.unit
class TestBoltonConstruction:
    """Mode validation and JIT function selection."""

    @pytest.mark.parametrize(
        "mode", [PotentialTemperatureMode.LCL, PotentialTemperatureMode.EQUIVALENT]
    )
    def test_supported_modes_construct(self, mode):
        equation = BoltonPotentialTempEquation(mode=mode)

        assert equation.mode is mode
        assert equation.name is ThermodynamicEquationName.POTENTIAL_TEMP

    def test_dry_mode_rejected(self):
        """DRY belongs to PoissonEquation, not the Bolton moist equations."""
        with pytest.raises(ValueError, match="does not support mode"):
            BoltonPotentialTempEquation(mode=PotentialTemperatureMode.DRY)

    def test_default_mode_is_equivalent(self):
        assert BoltonPotentialTempEquation().mode is PotentialTemperatureMode.EQUIVALENT

    def test_modes_dispatch_to_different_kernels(self, standard_parcel):
        """theta-L and theta-E must not resolve to the same JIT function."""
        result_l = BoltonPotentialTempEquation(
            mode=PotentialTemperatureMode.LCL
        ).calculate(**standard_parcel)
        result_e = BoltonPotentialTempEquation(
            mode=PotentialTemperatureMode.EQUIVALENT
        ).calculate(**standard_parcel)

        assert result_l != result_e
        assert result_e > result_l, "theta-E must exceed theta-L for a moist parcel"


@pytest.mark.unit
class TestBoltonDispatch:
    """Five-input broadcasting, scalar/array dispatch, shape preservation."""

    def test_scalar_returns_python_float(self, theta_e, standard_parcel):
        result = theta_e.calculate(**standard_parcel)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_array_returns_ndarray_same_shape(self, theta_e):
        inputs = {
            key: np.array(
                [
                    moist_parcel_inputs(t, d, p)[key]
                    for _, t, d, p in ATMOSPHERIC_PROFILES
                ]
            )
            for key in (
                "temp_k",
                "lcl_temp_k",
                "pressure_hpa",
                "vapor_pressure_hpa",
                "mixing_ratio",
            )
        }

        result = theta_e.calculate(**inputs)

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(ATMOSPHERIC_PROFILES),)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("shape", [(2, 3), (2, 2, 2)])
    def test_multidimensional_shape_preserved(self, theta_e, standard_parcel, shape):
        inputs = {k: np.full(shape, v) for k, v in standard_parcel.items()}

        result = theta_e.calculate(**inputs)

        assert result.shape == shape

    def test_partial_broadcast(self, theta_e, standard_parcel):
        """One array argument among scalars broadcasts the whole calculation."""
        inputs = dict(standard_parcel)
        inputs["mixing_ratio"] = np.array([0.005, 0.010, 0.013])

        result = theta_e.calculate(**inputs)

        assert result.shape == (3,)
        assert np.all(np.diff(result) > 0), "theta-E must increase with moisture"

    @pytest.mark.parametrize(
        "mode", [PotentialTemperatureMode.LCL, PotentialTemperatureMode.EQUIVALENT]
    )
    def test_scalar_and_vector_paths_agree(self, mode):
        """
        Regression guard for the p0 indexing bug: the vectorised kernel
        must reproduce the scalar kernel exactly.
        """
        equation = BoltonPotentialTempEquation(mode=mode)
        per_profile = [
            moist_parcel_inputs(t, d, p) for _, t, d, p in ATMOSPHERIC_PROFILES
        ]
        array_inputs = {
            key: np.array([profile[key] for profile in per_profile])
            for key in per_profile[0]
        }

        vector_result = equation.calculate(**array_inputs)
        scalar_result = np.array(
            [equation.calculate(**profile) for profile in per_profile]
        )

        np.testing.assert_allclose(vector_result, scalar_result, rtol=0, atol=TOL_EXACT)

    def test_custom_reference_pressure(self, theta_e, standard_parcel):
        result_1000 = theta_e.calculate(**standard_parcel, p0=1000.0)
        result_500 = theta_e.calculate(**standard_parcel, p0=500.0)

        assert result_500 < result_1000


@pytest.mark.unit
class TestBoltonPhysics:
    """Analytical limits and invariants for the moist equations."""

    def test_theta_l_reduces_to_poisson_in_dry_limit(self, poisson, theta_l):
        """
        As e -> 0 and w -> 0, Bolton eq. 24 collapses to the Poisson relation.
        """
        temp_k, pressure_hpa = 288.15, 850.0

        dry_theta = poisson.calculate(temp_k=temp_k, pressure_hpa=pressure_hpa)
        limit_theta_l = theta_l.calculate(
            temp_k=temp_k,
            lcl_temp_k=temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=0.0,
            mixing_ratio=0.0,
        )

        assert limit_theta_l == pytest.approx(dry_theta, abs=1e-9)

    def test_theta_e_converges_to_theta_l_in_dry_limit(self, theta_l, theta_e):
        """The exponential moisture term tends to 1 as w -> 0."""
        base = {
            "temp_k": 288.15,
            "lcl_temp_k": 288.15,
            "pressure_hpa": 850.0,
            "vapor_pressure_hpa": 0.0,
            "mixing_ratio": 1e-8,
        }

        assert theta_e.calculate(**base) == pytest.approx(
            theta_l.calculate(**base), abs=1e-3
        )

    @pytest.mark.parametrize(
        "label,temp_c,dewpoint_c,pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_ordering_theta_le_theta_l_le_theta_e(
        self, poisson, theta_l, theta_e, label, temp_c, dewpoint_c, pressure_hpa
    ):
        """
        Thermodynamic invariant for any unsaturated parcel: adding the
        moisture corrections can only raise the potential temperature.
        """
        parcel = moist_parcel_inputs(temp_c, dewpoint_c, pressure_hpa)

        dry = poisson.calculate(
            temp_k=parcel["temp_k"], pressure_hpa=parcel["pressure_hpa"]
        )
        moist_l = theta_l.calculate(**parcel)
        moist_e = theta_e.calculate(**parcel)

        assert dry <= moist_l + 1e-6, f"{label}: theta={dry} > theta_L={moist_l}"
        assert moist_l <= moist_e + 1e-6, (
            f"{label}: theta_L={moist_l} > theta_E={moist_e}"
        )

    def test_theta_e_increases_with_moisture(self, theta_e, standard_parcel):
        inputs = dict(standard_parcel)
        inputs["mixing_ratio"] = np.array([0.001, 0.005, 0.010, 0.020])

        results = theta_e.calculate(**inputs)

        assert np.all(np.diff(results) > 0)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestBoltonAgainstMetPy:
    """
    Cross-validation of theta-E against MetPy.

    MetPy implements Bolton (1980) eq. 43 with a closed-form LCL temperature,
    while meteocalc uses its Brent LCL solver, so small differences are
    expected and bounded by TOL_THETA_E_K.
    """

    @pytest.mark.parametrize(
        "label,temp_c,dewpoint_c,pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_matches_metpy_scalar(
        self, theta_e, label, temp_c, dewpoint_c, pressure_hpa
    ):
        temp_k = temp_c + 273.15
        dewpoint_k = dewpoint_c + 273.15

        ours = theta_e.calculate(
            **moist_parcel_inputs(temp_c, dewpoint_c, pressure_hpa)
        )
        reference = float(
            equivalent_potential_temperature(
                pressure_hpa * metpy_units.hPa,
                temp_k * metpy_units.K,
                dewpoint_k * metpy_units.K,
            )
            .to("K")
            .magnitude
        )

        assert ours == pytest.approx(reference, abs=TOL_THETA_E_K), (
            f"{label}: ours={ours:.4f} K metpy={reference:.4f} K "
            f"err={abs(ours - reference):.4f} K"
        )

    def test_matches_metpy_across_profile_array(self, theta_e):
        """Array path validated end to end against MetPy."""
        per_profile = [
            moist_parcel_inputs(t, d, p) for _, t, d, p in ATMOSPHERIC_PROFILES
        ]
        array_inputs = {
            key: np.array([profile[key] for profile in per_profile])
            for key in per_profile[0]
        }

        ours = theta_e.calculate(**array_inputs)
        reference = np.array(
            [
                float(
                    equivalent_potential_temperature(
                        p * metpy_units.hPa,
                        (t + 273.15) * metpy_units.K,
                        (d + 273.15) * metpy_units.K,
                    )
                    .to("K")
                    .magnitude
                )
                for _, t, d, p in ATMOSPHERIC_PROFILES
            ]
        )

        np.testing.assert_allclose(ours, reference, atol=TOL_THETA_E_K)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestMoistInputsAgainstPsychroLib:
    """
    Validate the vapour pressure and mixing ratio feeding theta-L/theta-E
    against PsychroLib's ASHRAE implementation.

    An error here would propagate into every moist potential temperature,
    so these inputs are validated independently of MetPy.
    """

    @pytest.mark.parametrize(
        "label,temp_c,dewpoint_c,pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_vapor_pressure_matches_ashrae(
        self, label, temp_c, dewpoint_c, pressure_hpa
    ):
        ours_pa = actual_vapor_pressure_hpa(dewpoint_c + 273.15) * 100.0
        reference_pa = psychrolib.GetSatVapPres(dewpoint_c)

        assert ours_pa == pytest.approx(reference_pa, abs=TOL_VAPOR_PA), (
            f"{label}: ours={ours_pa:.3f} Pa ashrae={reference_pa:.3f} Pa"
        )

    @pytest.mark.parametrize(
        "label,temp_c,dewpoint_c,pressure_hpa", ATMOSPHERIC_PROFILES, ids=PROFILE_IDS
    )
    def test_mixing_ratio_matches_ashrae(self, label, temp_c, dewpoint_c, pressure_hpa):
        ours = mixing_ratio_kgkg(dewpoint_c + 273.15, pressure_hpa)
        reference = psychrolib.GetHumRatioFromTDewPoint(
            dewpoint_c, pressure_hpa * 100.0
        )

        assert ours == pytest.approx(reference, abs=TOL_MIXING_RATIO), (
            f"{label}: ours={ours:.6f} ashrae={reference:.6f} kg/kg"
        )


# ============================================================================
# Edge cases
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Guard rails and boundary behaviour."""

    def test_theta_e_returns_nan_beyond_exponent_guard(self, theta_e, standard_parcel):
        """
        _bolton_theta_e_scalar returns NaN when the exponential term exceeds
        50 to avoid overflow. Requires a physically impossible mixing ratio.
        """
        inputs = dict(standard_parcel)
        inputs["mixing_ratio"] = 6.0

        assert np.isnan(theta_e.calculate(**inputs))

    def test_nan_guard_applies_elementwise_in_array_path(
        self, theta_e, standard_parcel
    ):
        """A single out-of-range element must not poison its neighbours."""
        inputs = dict(standard_parcel)
        inputs["mixing_ratio"] = np.array([0.010, 6.0, 0.013])

        result = theta_e.calculate(**inputs)

        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])

    def test_saturated_parcel_lcl_equals_temperature(self, theta_l, theta_e):
        """When T == Td the parcel is already saturated and T_LCL == T."""
        temp_k, pressure_hpa = 293.15, 1013.25
        vapor_pressure = actual_vapor_pressure_hpa(temp_k)
        mixing_ratio = eps * vapor_pressure / (pressure_hpa - vapor_pressure)
        inputs = {
            "temp_k": temp_k,
            "lcl_temp_k": temp_k,
            "pressure_hpa": pressure_hpa,
            "vapor_pressure_hpa": vapor_pressure,
            "mixing_ratio": mixing_ratio,
        }

        assert np.isfinite(theta_l.calculate(**inputs))
        assert np.isfinite(theta_e.calculate(**inputs))

    def test_single_element_array_stays_array(self, poisson):
        """A length-1 array is not a scalar and must return an array."""
        result = poisson.calculate(
            temp_k=np.array([288.15]), pressure_hpa=np.array([850.0])
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    @pytest.mark.parametrize(
        "temp_k,pressure_hpa",
        [
            (180.0, 100.0),
            (220.0, 50.0),
            (320.0, 1050.0),
        ],
    )
    def test_extreme_but_valid_conditions(self, poisson, temp_k, pressure_hpa):
        result = poisson.calculate(temp_k=temp_k, pressure_hpa=pressure_hpa)

        assert np.isfinite(result)
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
