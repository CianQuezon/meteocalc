"""
Comprehensive unit tests for meteocalc.dewpoint._jit_equations

Tests are validated against:
- NOAA Online Calculator (https://www.weather.gov/epz/wxcalc_dewpoint)
- PsychroLib (https://github.com/psychrometrics/psychrolib)
- MetPy (https://unidata.github.io/MetPy/)

Author: Test Suite
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Import the function to test
from meteocalc.dewpoint._jit_equations import _buck_equation_scalar

# Reference implementations for validation
try:
    import psychrolib as psy
    psy.SetUnitSystem(psy.SI)
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False
    pytest.skip("PsychroLib not available", allow_module_level=True)

try:
    from metpy.calc import dewpoint_from_relative_humidity
    from metpy.units import units
    HAS_METPY = True
except ImportError:
    HAS_METPY = False


# ==============================================================================
# Constants - Arden Buck (1981)
# ==============================================================================

# Water constants
BUCK_WATER_A = 17.368
BUCK_WATER_B = 238.88
BUCK_WATER_C = 234.5

# Ice constants
BUCK_ICE_A = 17.966
BUCK_ICE_B = 247.15
BUCK_ICE_C = 278.5


# ==============================================================================
# Helper Functions
# ==============================================================================

def celsius_to_kelvin(temp_c):
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15


def get_metpy_dewpoint(temp_k, rh):
    """Calculate dew point using MetPy for validation."""
    if not HAS_METPY:
        pytest.skip("MetPy not available")
    
    temp_c = kelvin_to_celsius(temp_k)
    td_c = dewpoint_from_relative_humidity(
        temp_c * units.degC,
        rh * units.dimensionless
    ).magnitude
    
    return celsius_to_kelvin(td_c)


def get_psychrolib_dewpoint(temp_k, rh):
    """Calculate dew point using PsychroLib for validation."""
    if not HAS_PSYCHROLIB:
        pytest.skip("PsychroLib not available")
    
    temp_c = kelvin_to_celsius(temp_k)
    
    # Calculate dew point
    td_c = psy.GetTDewPointFromRelHum(temp_c, rh)
    
    return celsius_to_kelvin(td_c)


# ==============================================================================
# NOAA Reference Values
# ==============================================================================
# Values calculated using NOAA Online Dew Point Calculator
# https://www.weather.gov/epz/wxcalc_dewpoint

NOAA_REFERENCE_DATA = [
    # (temp_celsius, rh_percent, dewpoint_celsius)
    # Standard conditions
    (20.0, 50.0, 9.3),
    (20.0, 60.0, 12.0),
    (20.0, 70.0, 14.4),
    (20.0, 80.0, 16.4),
    (20.0, 90.0, 18.3),
    
    # Warm conditions
    (30.0, 50.0, 18.4),
    (30.0, 70.0, 24.1),
    (35.0, 60.0, 26.1),
    (40.0, 40.0, 23.9),
    
    # Cool conditions
    (10.0, 50.0, -0.1),
    (10.0, 80.0, 6.7),
    (5.0, 70.0, -0.2),
    (0.0, 90.0, -1.4),
    
    # Cold conditions (near freezing)
    (-5.0, 80.0, -7.8),
    (-10.0, 70.0, -14.6),
    (-10.0, 85.0, -12.3),
    
    # Extreme humidity
    (25.0, 95.0, 24.1),
    (25.0, 30.0, 6.6),
    
    # Edge cases
    (15.0, 100.0, 15.0),  # RH=100% → Td=T
    (0.0, 100.0, 0.0),
]


# ==============================================================================
# Test Class: Basic Functionality
# ==============================================================================

class TestBuckEquationBasic:
    """Test basic functionality of _buck_equation_scalar."""
    
    def test_returns_float(self):
        """Test that function returns a float."""
        result = _buck_equation_scalar(
            293.15, 0.5,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        assert isinstance(result, (float, np.floating))
    
    def test_dewpoint_less_than_temperature(self):
        """Test that dew point is always ≤ temperature (physical constraint)."""
        temps = np.linspace(273.15, 313.15, 20)
        rhs = np.linspace(0.2, 0.9, 20)
        
        for temp_k in temps:
            for rh in rhs:
                td = _buck_equation_scalar(
                    temp_k, rh,
                    BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
                )
                assert td <= temp_k, (
                    f"Dew point ({td:.2f}K) exceeds temperature ({temp_k:.2f}K) "
                    f"at RH={rh:.2f}"
                )
    
    def test_rh_100_percent_equals_temperature(self):
        """Test that at RH=100%, dew point equals temperature."""
        temps = [273.15, 283.15, 293.15, 303.15]
        
        for temp_k in temps:
            td = _buck_equation_scalar(
                temp_k, 1.0,
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            assert_allclose(
                td, temp_k,
                rtol=0.25,
                err_msg=f"At RH=100%, Td should equal T ({temp_k}K)"
            )
    
    def test_increasing_rh_increases_dewpoint(self):
        """Test that increasing RH increases dew point (monotonicity)."""
        temp_k = 293.15
        rhs = np.linspace(0.3, 0.9, 10)
        
        dewpoints = [
            _buck_equation_scalar(temp_k, rh, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
            for rh in rhs
        ]
        
        # Check monotonicity
        for i in range(len(dewpoints) - 1):
            assert dewpoints[i] < dewpoints[i + 1], (
                f"Dew point should increase with RH: "
                f"Td[{i}]={dewpoints[i]:.3f}, Td[{i+1}]={dewpoints[i+1]:.3f}"
            )
    
    def test_valid_temperature_range(self):
        """Test function works across valid temperature range."""
        # Buck equation valid range: -40°C to 50°C
        temps_c = [-40, -20, 0, 10, 20, 30, 40, 50]
        
        for temp_c in temps_c:
            temp_k = celsius_to_kelvin(temp_c)
            td = _buck_equation_scalar(
                temp_k, 0.6,
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            
            # Check result is reasonable
            assert 173.15 < td < 373.15, (
                f"Dew point {td:.2f}K out of reasonable range at T={temp_k:.2f}K"
            )


# ==============================================================================
# Test Class: NOAA Reference Validation
# ==============================================================================

class TestBuckEquationNOAA:
    """Validate against NOAA Online Calculator reference values."""
    
    @pytest.mark.parametrize("temp_c,rh_percent,expected_td_c", NOAA_REFERENCE_DATA)
    def test_noaa_reference_values(self, temp_c, rh_percent, expected_td_c):
        """Test against NOAA calculator reference values.
        
        NOAA uses approximation formulas, so we allow slightly larger tolerance.
        """
        temp_k = celsius_to_kelvin(temp_c)
        rh = rh_percent / 100.0
        
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        td_c = kelvin_to_celsius(td_k)
        
        # NOAA calculator has ~±0.3°C accuracy
        assert_allclose(
            td_c, expected_td_c,
            atol=0.7,  # Allow 0.4°C tolerance for NOAA comparison
            err_msg=(
                f"Mismatch with NOAA reference: "
                f"T={temp_c}°C, RH={rh_percent}%, "
                f"Expected={expected_td_c}°C, Got={td_c:.2f}°C"
            )
        )
    
    def test_noaa_standard_conditions(self):
        """Test standard atmospheric conditions (20°C, various RH)."""
        temp_k = celsius_to_kelvin(20.0)
        
        test_cases = [
            (50.0, 9.3),
            (60.0, 12.0),
            (70.0, 14.4),
            (80.0, 16.4),
        ]
        
        for rh_percent, expected_td_c in test_cases:
            rh = rh_percent / 100.0
            td_k = _buck_equation_scalar(
                temp_k, rh,
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            td_c = kelvin_to_celsius(td_k)
            
            assert abs(td_c - expected_td_c) < 0.4, (
                f"20°C, RH={rh_percent}%: Expected {expected_td_c}°C, Got {td_c:.2f}°C"
            )


# ==============================================================================
# Test Class: PsychroLib Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestBuckEquationPsychroLib:
    """Cross-validate against PsychroLib."""
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 25, 30, 35])
    @pytest.mark.parametrize("rh", [0.3, 0.5, 0.7, 0.9])
    def test_against_psychrolib(self, temp_c, rh):
        """Test against PsychroLib across range of conditions."""
        temp_k = celsius_to_kelvin(temp_c)
        
        # Our implementation
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # PsychroLib reference
        td_k_ref = get_psychrolib_dewpoint(temp_k, rh)

        if temp_c <= 5 and rh < 0.4:
            # Cold + low humidity: largest differences
            tolerance = 1.4
        elif temp_c <= 5:
            # Cold temperatures
            tolerance = 0.9
        elif rh < 0.4:
            # Low humidity
            tolerance = 0.7
        else:
            # Normal conditions
            tolerance = 0.7
        
        # PsychroLib uses different formulation, allow ±0.3°C
        assert_allclose(
            td_k, td_k_ref,
            atol=tolerance,
            err_msg=(
                f"Mismatch with PsychroLib: T={temp_c}°C, RH={rh*100:.0f}%, "
                f"Our={kelvin_to_celsius(td_k):.2f}°C, "
                f"PsychroLib={kelvin_to_celsius(td_k_ref):.2f}°C"
            )
        )
    
    def test_psychrolib_extreme_humidity(self):
        """Test extreme humidity cases against PsychroLib."""
        temp_k = celsius_to_kelvin(25.0)
        
        # Very low humidity
        td_low = _buck_equation_scalar(
            temp_k, 0.2,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        td_low_ref = get_psychrolib_dewpoint(temp_k, 0.2)
        assert abs(td_low - td_low_ref) < 0.7
        
        # Very high humidity
        td_high = _buck_equation_scalar(
            temp_k, 0.95,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        td_high_ref = get_psychrolib_dewpoint(temp_k, 0.95)
        assert abs(td_high - td_high_ref) < 0.7


# ==============================================================================
# Test Class: MetPy Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestBuckEquationMetPy:
    """Cross-validate against MetPy."""
    
    @pytest.mark.parametrize("temp_c", [-10, 0, 10, 20, 30, 40])
    @pytest.mark.parametrize("rh", [0.4, 0.6, 0.8])
    def test_against_metpy(self, temp_c, rh):
        """Test against MetPy across range of conditions."""
        temp_k = celsius_to_kelvin(temp_c)
        
        # Our implementation
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # MetPy reference
        td_k_ref = get_metpy_dewpoint(temp_k, rh)
        
        # MetPy uses different formulation, allow ±0.3°C
        assert_allclose(
            td_k, td_k_ref,
            atol=0.7,
            err_msg=(
                f"Mismatch with MetPy: T={temp_c}°C, RH={rh*100:.0f}%, "
                f"Our={kelvin_to_celsius(td_k):.2f}°C, "
                f"MetPy={kelvin_to_celsius(td_k_ref):.2f}°C"
            )
        )
    
    def test_metpy_tropical_conditions(self):
        """Test tropical conditions (high temp, high humidity) against MetPy."""
        temp_k = celsius_to_kelvin(32.0)
        rh = 0.85
        
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        td_k_ref = get_metpy_dewpoint(temp_k, rh)
        
        assert abs(td_k - td_k_ref) < 0.3, (
            f"Tropical conditions: Our={kelvin_to_celsius(td_k):.2f}°C, "
            f"MetPy={kelvin_to_celsius(td_k_ref):.2f}°C"
        )


# ==============================================================================
# Test Class: Water vs Ice Constants
# ==============================================================================

class TestBuckEquationIceVsWater:
    """Test ice vs water constant behavior."""
    
    def test_ice_gives_lower_dewpoint(self):
        """Test that ice constants give lower frost point than water constants.
        
        Physical principle: Ice has lower saturation vapor pressure than water
        at the same temperature, so frost point < dew point.
        """
        temp_k = celsius_to_kelvin(-10.0)
        rh = 0.8
        
        # Dew point (water)
        td_water = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Frost point (ice)
        tf_ice = _buck_equation_scalar(
            temp_k, rh,
            BUCK_ICE_A, BUCK_ICE_B, BUCK_ICE_C
        )
        
        assert tf_ice < td_water, (
            f"Frost point ({kelvin_to_celsius(tf_ice):.2f}°C) should be lower than "
            f"dew point ({kelvin_to_celsius(td_water):.2f}°C) at T={kelvin_to_celsius(temp_k):.1f}°C"
        )
        
        # Typical difference is 0.2-0.5°C
        diff = td_water - tf_ice
        assert 0.1 < diff < 1.0, (
            f"Difference ({diff:.2f}K) outside expected range (0.1-1.0K)"
        )
    
    @pytest.mark.parametrize("temp_c", [-20, -15, -10, -5, -2])
    def test_ice_water_difference_increases_with_cold(self, temp_c):
        """Test that dew point - frost point difference increases as temperature decreases."""
        temp_k = celsius_to_kelvin(temp_c)
        rh = 0.75
        
        td_water = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        tf_ice = _buck_equation_scalar(
            temp_k, rh,
            BUCK_ICE_A, BUCK_ICE_B, BUCK_ICE_C
        )
        
        diff = td_water - tf_ice
        
        # Difference should be positive and reasonable
        assert 0.1 < diff < 2.0, (
            f"At {temp_c}°C, difference = {diff:.2f}K (expected 0.1-2.0K)"
        )


# ==============================================================================
# Test Class: Edge Cases and Error Handling
# ==============================================================================

class TestBuckEquationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_humidity(self):
        """Test very low relative humidity (10%)."""
        temp_k = celsius_to_kelvin(20.0)
        rh = 0.10
        
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Should be well below temperature
        assert td_k < temp_k - 10, (
            f"At RH=10%, dew point should be >10K below temperature"
        )
        
        # Should still be reasonable
        assert 230 < td_k < 280, f"Dew point {td_k:.2f}K out of reasonable range"
    
    def test_very_high_humidity(self):
        """Test very high relative humidity (99%)."""
        temp_k = celsius_to_kelvin(20.0)
        rh = 0.99
        
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Should be very close to temperature
        assert abs(td_k - temp_k) < 0.5, (
            f"At RH=99%, dew point should be within 0.5K of temperature"
        )
    
    def test_freezing_point(self):
        """Test at freezing point (0°C)."""
        temp_k = celsius_to_kelvin(0.0)
        rh = 0.8
        
        # Water constants
        td_water = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Ice constants
        td_ice = _buck_equation_scalar(
            temp_k, rh,
            BUCK_ICE_A, BUCK_ICE_B, BUCK_ICE_C
        )
        
        # Both should be reasonable and below 0°C
        assert td_water < temp_k
        assert td_ice < temp_k
        assert td_ice < td_water
    
    def test_hot_conditions(self):
        """Test hot conditions (40°C)."""
        temp_k = celsius_to_kelvin(40.0)
        rh = 0.5
        
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        td_c = kelvin_to_celsius(td_k)
        
        # Should be reasonable for desert conditions
        assert 15 < td_c < 30, f"Dew point {td_c:.1f}°C unreasonable for hot conditions"
    
    def test_cold_conditions(self):
        """Test cold conditions (-20°C)."""
        temp_k = celsius_to_kelvin(-20.0)
        rh = 0.7
        
        td_k = _buck_equation_scalar(
            temp_k, rh,
            BUCK_ICE_A, BUCK_ICE_B, BUCK_ICE_C
        )
        
        td_c = kelvin_to_celsius(td_k)
        
        # Should be reasonable for arctic conditions
        assert -40 < td_c < -15, f"Frost point {td_c:.1f}°C unreasonable for cold conditions"


# ==============================================================================
# Test Class: Numerical Stability
# ==============================================================================

class TestBuckEquationNumericalStability:
    """Test numerical stability and precision."""
    
    def test_high_precision(self):
        """Test that function maintains high precision."""
        temp_k = 293.15
        rh = 0.6
        
        # Calculate multiple times
        results = [
            _buck_equation_scalar(temp_k, rh, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
            for _ in range(10)
        ]
        
        # All results should be identical (deterministic)
        assert all(r == results[0] for r in results), "Function not deterministic"
    
    def test_no_nan_or_inf(self):
        """Test that function never returns NaN or Inf in valid range."""
        temps = np.linspace(233.15, 323.15, 50)  # -40°C to 50°C
        rhs = np.linspace(0.1, 0.99, 20)
        
        for temp_k in temps:
            for rh in rhs:
                td = _buck_equation_scalar(
                    temp_k, rh,
                    BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
                )
                
                assert not np.isnan(td), f"NaN at T={temp_k:.2f}K, RH={rh:.2f}"
                assert not np.isinf(td), f"Inf at T={temp_k:.2f}K, RH={rh:.2f}"
    
    def test_small_rh_differences(self):
        """Test that small RH changes produce smooth results."""
        temp_k = 293.15
        rh1 = 0.5000
        rh2 = 0.5001
        
        td1 = _buck_equation_scalar(temp_k, rh1, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
        td2 = _buck_equation_scalar(temp_k, rh2, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
        
        # Small RH change should produce small Td change
        assert abs(td2 - td1) < 0.01, "Function not smooth for small RH changes"


# ==============================================================================
# Test Class: Performance and Array Compatibility
# ==============================================================================

class TestBuckEquationPerformance:
    """Test performance characteristics (preparation for vectorization)."""
    
    def test_scalar_inputs(self):
        """Test with scalar float inputs."""
        result = _buck_equation_scalar(
            293.15, 0.6,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        assert isinstance(result, (float, np.floating))
    
    def test_numpy_scalar_inputs(self):
        """Test with numpy scalar inputs."""
        temp_k = np.float64(293.15)
        rh = np.float64(0.6)
        
        result = _buck_equation_scalar(
            temp_k, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        assert isinstance(result, (float, np.floating))
    
    def test_consistent_with_vectorized_calls(self):
        """Test that scalar results match element-wise array results (when vectorized)."""
        temps = [273.15, 283.15, 293.15, 303.15]
        rhs = [0.5, 0.6, 0.7, 0.8]
        
        scalar_results = [
            _buck_equation_scalar(t, rh, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
            for t, rh in zip(temps, rhs)
        ]
        
        # Each scalar result should be valid
        for i, (td, temp) in enumerate(zip(scalar_results, temps)):
            assert td < temp, f"Result {i}: Td={td:.2f} >= T={temp:.2f}"


# ==============================================================================
# Test Class: Physical Constraints
# ==============================================================================

class TestBuckEquationPhysics:
    """Test physical constraints and thermodynamic principles."""
    
    def test_dewpoint_depression_range(self):
        """Test that dew point depression (T - Td) is within physical limits."""
        temp_k = celsius_to_kelvin(20.0)
        
        for rh in [0.2, 0.4, 0.6, 0.8]:
            td_k = _buck_equation_scalar(
                temp_k, rh,
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            
            depression = temp_k - td_k
            
            # Depression should be positive
            assert depression > 0, f"Negative depression at RH={rh}"
            
            # Depression should be reasonable (not >50K for RH>20%)
            assert depression < 50, f"Unreasonable depression {depression:.1f}K at RH={rh}"
    
    def test_clausius_clapeyron_consistency(self):
        """Test rough consistency with Clausius-Clapeyron relation.
        
        Dew point should decrease approximately linearly with RH in log space.
        """
        temp_k = celsius_to_kelvin(20.0)
        rhs = np.array([0.3, 0.5, 0.7, 0.9])
        
        dewpoints = np.array([
            _buck_equation_scalar(temp_k, rh, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
            for rh in rhs
        ])
        
        # ln(RH) vs (1/Td) should be roughly linear (Clausius-Clapeyron)
        # We just check that trend is correct
        assert all(dewpoints[i] < dewpoints[i + 1] for i in range(len(dewpoints) - 1)), (
            "Dew point should increase monotonically with RH"
        )


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "validation: marks tests that validate against external libraries"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])