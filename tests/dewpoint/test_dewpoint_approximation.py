"""
Comprehensive test suite for Dewpoint.get_dewpoint_approximation()

Tests dew point approximation methods against reference implementations
from NOAA, MetPy, and PsychroLib to ensure accuracy and correctness.
"""

import pytest
import numpy as np
import warnings

# Import the function to test
from meteocalc.dewpoint.core import Dewpoint
from meteocalc.dewpoint._enums import DewPointEquationName

# Import reference implementations
try:
    from metpy.calc import dewpoint_from_relative_humidity
    from metpy.units import units
    HAS_METPY = True
except ImportError:
    HAS_METPY = False
    warnings.warn("MetPy not installed - skipping MetPy validation tests")

try:
    import psychrolib as psy
    psy.SetUnitSystem(psy.SI)
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False
    warnings.warn("PsychroLib not installed - skipping PsychroLib validation tests")


# NOAA reference values (from NOAA online calculator)
# https://www.weather.gov/epz/wxcalc_dewpoint
NOAA_REFERENCE_VALUES = [
    # (temp_c, rh, expected_td_c, tolerance)
    (20.0, 0.60, 12.0, 0.5),   # Standard conditions
    (30.0, 0.80, 26.2, 0.5),   # Hot and humid
    (10.0, 0.50, 0.0, 0.5),    # Cool, moderate RH
    (0.0, 0.70, -4.8, 0.5),    # At freezing
    (25.0, 0.40, 10.5, 0.5),   # Warm, dry
    (15.0, 0.90, 13.4, 0.5),   # Cool, very humid
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def scalar_conditions():
    """Standard scalar test conditions."""
    return {
        'temp_k': 293.15,  # 20°C
        'rh': 0.6,
        'expected_td_c': 12.0,
    }


@pytest.fixture
def array_conditions():
    """Standard array test conditions."""
    return {
        'temps_k': np.array([273.15, 283.15, 293.15, 303.15]),  # 0, 10, 20, 30°C
        'rhs': np.array([0.5, 0.6, 0.7, 0.8]),
        'expected_tds_c': np.array([-9.18, 2.60, 14.37, 26.17]),
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestBasicFunctionality:
    """Test basic function behavior and return types."""

    def test_scalar_input_returns_scalar(self, scalar_conditions):
        """Scalar inputs should return scalar float."""
        td = Dewpoint.get_dewpoint_approximation(
            temp_k=scalar_conditions['temp_k'],
            rh=scalar_conditions['rh']
        )

        assert isinstance(td, (float, np.floating)), \
            f"Expected float, got {type(td)}"
        assert np.isfinite(td), "Result should be finite"
        assert td > 0, "Temperature in Kelvin should be positive"

    def test_array_input_returns_array(self, array_conditions):
        """Array inputs should return ndarray with correct shape."""
        td = Dewpoint.get_dewpoint_approximation(
            temp_k=array_conditions['temps_k'],
            rh=0.6
        )

        assert isinstance(td, np.ndarray), \
            f"Expected ndarray, got {type(td)}"
        assert td.shape == array_conditions['temps_k'].shape
        assert np.all(np.isfinite(td))

    def test_list_input_works(self):
        """Lists should be converted and work correctly."""
        temps = [273.15, 283.15, 293.15]
        td = Dewpoint.get_dewpoint_approximation(temp_k=temps, rh=0.6)

        assert isinstance(td, np.ndarray)
        assert len(td) == len(temps)
        assert np.all(np.isfinite(td))


# ============================================================================
# NOAA Reference Validation Tests
# ============================================================================

class TestNOAAValidation:
    """Validate against NOAA calculator reference values."""

    @pytest.mark.parametrize("temp_c,rh,expected_td_c,tolerance", NOAA_REFERENCE_VALUES)
    def test_against_noaa_values(self, temp_c, rh, expected_td_c, tolerance):
        """Test against NOAA online calculator reference values."""
        temp_k = temp_c + 273.15

        td_k = Dewpoint.get_dewpoint_approximation(temp_k, rh)
        td_c = td_k - 273.15

        error = abs(td_c - expected_td_c)
        assert error < tolerance, \
            f"NOAA validation failed at {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  Expected: {expected_td_c:.2f}°C (NOAA)\n" \
            f"  Got:      {td_c:.2f}°C\n" \
            f"  Error:    {error:.3f}°C (tolerance: {tolerance}°C)"

    def test_noaa_array_validation(self, array_conditions):
        """Validate array calculation against NOAA values."""
        td_k = Dewpoint.get_dewpoint_approximation(
            temp_k=array_conditions['temps_k'],
            rh=array_conditions['rhs']
        )
        td_c = td_k - 273.15

        errors = np.abs(td_c - array_conditions['expected_tds_c'])
        max_error = np.max(errors)

        assert max_error < 0.5, \
            f"NOAA array validation failed:\n" \
            f"  Expected: {array_conditions['expected_tds_c']}\n" \
            f"  Got:      {np.round(td_c, 2)}\n" \
            f"  Max error: {max_error:.3f}°C"


# ============================================================================
# MetPy Validation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestMetPyValidation:
    """Validate against MetPy implementation."""

    def test_against_metpy_scalar(self, scalar_conditions):
        """Compare with MetPy for scalar input."""
        temp_k = scalar_conditions['temp_k']
        rh = scalar_conditions['rh']

        # Our implementation
        td_ours_k = Dewpoint.get_dewpoint_approximation(temp_k, rh)
        td_ours_c = td_ours_k - 273.15

        # MetPy implementation
        temp_metpy = (temp_k * units.K).to('degC')
        rh_metpy = rh * 100 * units.percent
        td_metpy_c = dewpoint_from_relative_humidity(
            temp_metpy, rh_metpy
        ).magnitude

        error = abs(td_ours_c - td_metpy_c)
        assert error < 0.5, \
            f"MetPy validation failed:\n" \
            f"  Ours:  {td_ours_c:.4f}°C\n" \
            f"  MetPy: {td_metpy_c:.4f}°C\n" \
            f"  Error: {error:.4f}°C"

    def test_against_metpy_array(self, array_conditions):
        """Compare with MetPy for array inputs."""
        temps_k = array_conditions['temps_k']
        rhs = array_conditions['rhs']

        # Our implementation
        td_ours_k = Dewpoint.get_dewpoint_approximation(temps_k, rhs)
        td_ours_c = td_ours_k - 273.15

        # MetPy implementation element-wise
        td_metpy_c = np.array([
            dewpoint_from_relative_humidity(
                (t * units.K).to('degC'),
                r * 100 * units.percent
            ).magnitude
            for t, r in zip(temps_k, rhs)
        ])

        errors = np.abs(td_ours_c - td_metpy_c)
        max_error = np.max(errors)

        assert max_error < 0.5, \
            f"MetPy array validation failed:\n" \
            f"  Ours:      {np.round(td_ours_c, 4)}\n" \
            f"  MetPy:     {np.round(td_metpy_c, 4)}\n" \
            f"  Max error: {max_error:.4f}°C"

    @pytest.mark.parametrize("temp_c,rh", [
        (20.0, 0.6),
        (30.0, 0.8),
        (10.0, 0.5),
        (0.0, 0.7),
        (25.0, 0.4),
    ])
    def test_against_metpy_parametrized(self, temp_c, rh):
        """Parametrized comparison with MetPy."""
        temp_k = temp_c + 273.15

        td_ours_c = Dewpoint.get_dewpoint_approximation(temp_k, rh) - 273.15

        td_metpy_c = dewpoint_from_relative_humidity(
            (temp_k * units.K).to('degC'),
            rh * 100 * units.percent
        ).magnitude

        error = abs(td_ours_c - td_metpy_c)
        assert error < 0.5, \
            f"At {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  Ours:  {td_ours_c:.4f}°C\n" \
            f"  MetPy: {td_metpy_c:.4f}°C\n" \
            f"  Error: {error:.4f}°C"


# ============================================================================
# PsychroLib Validation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestPsychroLibValidation:
    """Validate against PsychroLib implementation."""

    def test_against_psychrolib_scalar(self, scalar_conditions):
        """Compare with PsychroLib for scalar input."""
        temp_k = scalar_conditions['temp_k']
        rh = scalar_conditions['rh']
        temp_c = temp_k - 273.15

        # Our implementation
        td_ours_c = Dewpoint.get_dewpoint_approximation(temp_k, rh) - 273.15

        # PsychroLib implementation
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        error = abs(td_ours_c - td_psychro_c)
        assert error < 0.5, \
            f"PsychroLib validation failed:\n" \
            f"  Ours:       {td_ours_c:.4f}°C\n" \
            f"  PsychroLib: {td_psychro_c:.4f}°C\n" \
            f"  Error:      {error:.4f}°C"

    def test_against_psychrolib_array(self, array_conditions):
        """Compare with PsychroLib for array inputs."""
        temps_k = array_conditions['temps_k']
        rhs = array_conditions['rhs']

        # Our implementation
        td_ours_c = Dewpoint.get_dewpoint_approximation(temps_k, rhs) - 273.15

        # PsychroLib element-wise
        td_psychro_c = np.array([
            psy.GetTDewPointFromRelHum(t - 273.15, r)
            for t, r in zip(temps_k, rhs)
        ])

        errors = np.abs(td_ours_c - td_psychro_c)
        max_error = np.max(errors)

        assert max_error < 0.5, \
            f"PsychroLib array validation failed:\n" \
            f"  Ours:       {np.round(td_ours_c, 4)}\n" \
            f"  PsychroLib: {np.round(td_psychro_c, 4)}\n" \
            f"  Max error:  {max_error:.4f}°C"

    @pytest.mark.parametrize("temp_c,rh", [
        (20.0, 0.6),
        (30.0, 0.8),
        (10.0, 0.5),
        (0.0, 0.7),
        (25.0, 0.4),
    ])
    def test_against_psychrolib_parametrized(self, temp_c, rh):
        """Parametrized comparison with PsychroLib."""
        temp_k = temp_c + 273.15

        td_ours_c = Dewpoint.get_dewpoint_approximation(temp_k, rh) - 273.15
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        error = abs(td_ours_c - td_psychro_c)
        assert error < 0.5, \
            f"At {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  Ours:       {td_ours_c:.4f}°C\n" \
            f"  PsychroLib: {td_psychro_c:.4f}°C\n" \
            f"  Error:      {error:.4f}°C"


# ============================================================================
# Cross-Reference Validation
# ============================================================================

@pytest.mark.skipif(
    not (HAS_METPY and HAS_PSYCHROLIB),
    reason="Both MetPy and PsychroLib required"
)
class TestCrossReference:
    """Cross-validate between all three implementations."""

    @pytest.mark.parametrize("temp_c,rh", [
        (20.0, 0.6),
        (30.0, 0.8),
        (10.0, 0.5),
        (0.0, 0.7),
        (25.0, 0.4),
    ])
    def test_all_three_agree(self, temp_c, rh):
        """All three implementations should agree within tolerance."""
        temp_k = temp_c + 273.15

        # Our implementation
        td_ours_c = Dewpoint.get_dewpoint_approximation(temp_k, rh) - 273.15

        # MetPy
        td_metpy_c = dewpoint_from_relative_humidity(
            (temp_k * units.K).to('degC'),
            rh * 100 * units.percent
        ).magnitude

        # PsychroLib
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        # All three should agree within 0.5°C
        assert abs(td_ours_c - td_metpy_c) < 0.5, \
            f"Ours vs MetPy: {abs(td_ours_c - td_metpy_c):.4f}°C"

        assert abs(td_ours_c - td_psychro_c) < 0.5, \
            f"Ours vs PsychroLib: {abs(td_ours_c - td_psychro_c):.4f}°C"

        assert abs(td_metpy_c - td_psychro_c) < 0.5, \
            f"MetPy vs PsychroLib: {abs(td_metpy_c - td_psychro_c):.4f}°C"

    def test_all_three_agree_on_trend(self):
        """All three should show same trend with increasing RH."""
        temp_k = 293.15
        rhs = np.array([0.3, 0.5, 0.7, 0.9])

        # Our implementation
        td_ours = Dewpoint.get_dewpoint_approximation(temp_k, rhs) - 273.15

        # MetPy
        td_metpy = np.array([
            dewpoint_from_relative_humidity(
                (temp_k * units.K).to('degC'),
                rh * 100 * units.percent
            ).magnitude
            for rh in rhs
        ])

        # PsychroLib
        td_psychro = np.array([
            psy.GetTDewPointFromRelHum(temp_k - 273.15, rh)
            for rh in rhs
        ])

        # All should show increasing dew point with increasing RH
        assert np.all(np.diff(td_ours) > 0), \
            "Ours: dew point should increase with RH"
        assert np.all(np.diff(td_metpy) > 0), \
            "MetPy: dew point should increase with RH"
        assert np.all(np.diff(td_psychro) > 0), \
            "PsychroLib: dew point should increase with RH"


# ============================================================================
# Equation Selection Tests
# ============================================================================

class TestEquationSelection:
    """Test different equation selection methods."""

    def test_magnus_is_default(self):
        """Magnus should be the default equation."""
        td_default = Dewpoint.get_dewpoint_approximation(293.15, 0.6)
        td_magnus = Dewpoint.get_dewpoint_approximation(
            293.15, 0.6, dewpoint_equation_name='magnus'
        )

        assert td_default == td_magnus

    def test_enum_input_works(self):
        """Enum input should give same result as string."""
        td_string = Dewpoint.get_dewpoint_approximation(293.15, 0.6, 'magnus')
        td_enum = Dewpoint.get_dewpoint_approximation(
            293.15, 0.6, DewPointEquationName.MAGNUS
        )

        assert td_string == td_enum


# ============================================================================
# Input Validation Tests
# ============================================================================

class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_enum_string_raises(self):
        """Completely invalid string should raise ValueError from enum parsing."""
        with pytest.raises(ValueError, match="Invalid enum"):
            Dewpoint.get_dewpoint_approximation(
                293.15, 0.6, dewpoint_equation_name='invalid'
            )

    def test_solver_method_raises(self):
        """Solver method passed to approximation should raise ValueError."""
        with pytest.raises(ValueError, match="not a valid approximation"):
            Dewpoint.get_dewpoint_approximation(
                293.15, 0.6, dewpoint_equation_name='vapor_inversion'
            )

    def test_rh_below_zero_raises(self):
        """Negative RH should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_dewpoint_approximation(293.15, -0.1)

    def test_rh_above_one_raises(self):
        """RH > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_dewpoint_approximation(293.15, 1.5)

    def test_temperature_out_of_range_warns(self):
        """Temperature outside valid range should warn."""
        with pytest.warns(UserWarning, match="outside valid range"):
            Dewpoint.get_dewpoint_approximation(400.0, 0.6)


# ============================================================================
# Broadcasting Tests
# ============================================================================

class TestBroadcasting:
    """Test broadcasting between scalar and array inputs."""

    def test_scalar_temp_array_rh(self):
        """Scalar temp with array RH should broadcast."""
        rhs = np.array([0.5, 0.6, 0.7, 0.8])
        td = Dewpoint.get_dewpoint_approximation(temp_k=293.15, rh=rhs)

        assert td.shape == rhs.shape
        assert np.all(np.isfinite(td))

    def test_array_temp_scalar_rh(self):
        """Array temp with scalar RH should broadcast."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        td = Dewpoint.get_dewpoint_approximation(temp_k=temps, rh=0.6)

        assert td.shape == temps.shape
        assert np.all(np.isfinite(td))

    def test_both_arrays_same_shape(self):
        """Both arrays with same shape should work."""
        temps = np.array([273.15, 283.15, 293.15])
        rhs = np.array([0.5, 0.6, 0.7])
        td = Dewpoint.get_dewpoint_approximation(temp_k=temps, rh=rhs)

        assert td.shape == temps.shape
        assert np.all(np.isfinite(td))

    def test_incompatible_shapes_raises(self):
        """Arrays with incompatible shapes should raise."""
        temps = np.array([273.15, 283.15, 293.15])
        rhs = np.array([0.5, 0.6])

        with pytest.raises(ValueError, match="same shape"):
            Dewpoint.get_dewpoint_approximation(temp_k=temps, rh=rhs)

    def test_2d_array_preserves_shape(self):
        """2D array inputs should preserve shape."""
        temps_2d = np.array([[273.15, 283.15], [293.15, 303.15]])
        td = Dewpoint.get_dewpoint_approximation(temp_k=temps_2d, rh=0.6)

        assert td.shape == temps_2d.shape


# ============================================================================
# Physical Correctness Tests
# ============================================================================

class TestPhysicalCorrectness:
    """Test physical relationships and constraints."""

    def test_dewpoint_below_air_temp(self):
        """Dew point must be at or below air temperature."""
        temp_k = 293.15
        td = Dewpoint.get_dewpoint_approximation(temp_k, 0.6)

        assert td <= temp_k, \
            f"Dew point ({td}K) must be <= air temp ({temp_k}K)"

    def test_increases_with_rh(self):
        """Higher RH should give higher dew point."""
        temp_k = 293.15
        td_low = Dewpoint.get_dewpoint_approximation(temp_k, 0.3)
        td_high = Dewpoint.get_dewpoint_approximation(temp_k, 0.9)

        assert td_high > td_low, \
            f"Higher RH should give higher dew point: {td_high} vs {td_low}"

    def test_saturation_equals_temp(self):
        """At 100% RH, dew point should equal air temperature."""
        temp_k = 293.15
        td = Dewpoint.get_dewpoint_approximation(temp_k, 1.0)

        assert abs(td - temp_k) < 0.01, \
            f"At saturation, dew point ({td}K) should equal temp ({temp_k}K)"

    def test_increases_with_temp(self):
        """At same RH, warmer air should give higher dew point."""
        rh = 0.6
        td_cold = Dewpoint.get_dewpoint_approximation(273.15, rh)
        td_warm = Dewpoint.get_dewpoint_approximation(303.15, rh)

        assert td_warm > td_cold

    def test_monotonically_increasing_with_rh(self):
        """Dew point should increase monotonically with RH."""
        temp_k = 293.15
        rhs = np.linspace(0.1, 1.0, 20)
        tds = Dewpoint.get_dewpoint_approximation(temp_k, rhs)

        assert np.all(np.diff(tds) > 0), \
            "Dew point should increase monotonically with RH"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with other meteocalc components."""

    def test_dew_point_below_frost_point(self):
        """Below 0°C, dew point should be less than frost point."""
        temp_k = 263.15  # -10°C
        rh = 0.7

        td = Dewpoint.get_dewpoint_approximation(temp_k, rh)
        tf = Dewpoint.get_frostpoint_approximation(temp_k, rh)

        assert td < tf, \
            f"Dew point ({td}K) should be < frost point ({tf}K) below 0°C"

    def test_consistency_with_solver(self):
        """Approximation should be close to exact solver."""
        temp_k = 293.15
        rh = 0.6

        td_approx = Dewpoint.get_dewpoint_approximation(temp_k, rh)
        td_exact = Dewpoint.get_dewpoint_solver(temp_k, rh)

        diff = abs(td_approx - td_exact)
        assert diff < 0.5, \
            f"Approximation differs from solver by {diff:.3f}K (expected <0.5K)"

    def test_convenience_wrapper_consistent(self):
        """Convenience wrapper should give same result as explicit call."""
        temp_k = 293.15
        rh = 0.6

        td_explicit = Dewpoint.get_dewpoint_approximation(temp_k, rh)
        td_wrapper = Dewpoint.get_dewpoint(
            temp_k, rh, calculation_method='approximation'
        )

        assert td_explicit == td_wrapper, \
            "Explicit and wrapper should give identical results"


# ============================================================================
# Summary Report
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])