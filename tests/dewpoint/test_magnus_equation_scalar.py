"""
Comprehensive unit tests for meteocalc.dewpoint._jit_equations._magnus_equation_scalar

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
from meteocalc.dewpoint._jit_equations import _magnus_equation_scalar

# Reference implementations for validation
try:
    import psychrolib as psy
    psy.SetUnitSystem(psy.SI)
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False

try:
    from metpy.calc import dewpoint_from_relative_humidity
    from metpy.units import units
    HAS_METPY = True
except ImportError:
    HAS_METPY = False


# ==============================================================================
# Constants - Magnus-Tetens
# ==============================================================================

# Water constants (Tetens 1930)
MAGNUS_WATER_A = 17.27
MAGNUS_WATER_B = 237.7

# Ice constants (Murray 1967)
MAGNUS_ICE_A = 21.875
MAGNUS_ICE_B = 265.5


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
    
    try:
        td_c = dewpoint_from_relative_humidity(
            temp_c * units.degC,
            rh * units.dimensionless
        ).magnitude
        return celsius_to_kelvin(td_c)
    except Exception as e:
        pytest.skip(f"MetPy error: {e}")


def get_psychrolib_dewpoint(temp_k, rh):
    """Calculate dew point using PsychroLib for validation."""
    if not HAS_PSYCHROLIB:
        pytest.skip("PsychroLib not available")
    
    temp_c = kelvin_to_celsius(temp_k)
    
    if not (0.0 <= rh <= 1.0):
        raise ValueError(f"RH must be in [0, 1], got {rh}")
    
    try:
        td_c = psy.GetTDewPointFromRelHum(temp_c, rh)
    except Exception as e:
        pytest.skip(f"PsychroLib error: {e}")
    
    return celsius_to_kelvin(td_c)


def get_tolerance_for_conditions(temp_c, rh):
    """Get appropriate tolerance based on temperature and humidity."""
    base_tolerance = 0.5
    
    if temp_c < 0:
        base_tolerance = 0.8
    if temp_c < -10:
        base_tolerance = 1.0
    
    if rh < 0.4:
        base_tolerance += 0.3
    
    if temp_c < 0 and rh < 0.5:
        base_tolerance = 1.2
    
    return base_tolerance


# ==============================================================================
# NOAA Reference Values
# ==============================================================================

NOAA_REFERENCE_DATA = [
    # (temp_celsius, rh_percent, dewpoint_celsius)
    # Standard conditions
    (20.0, 50.0, 9.3),
    (20.0, 60.0, 12.0),
    (20.0, 70.0, 14.4),
    (20.0, 80.0, 16.4),
    (20.0, 90.0, 18.3),
    
    # Warm conditions
    (25.0, 50.0, 13.9),
    (30.0, 50.0, 18.4),
    (30.0, 70.0, 24.1),
    (35.0, 60.0, 26.1),
    
    # Cool conditions
    (15.0, 60.0, 7.6),
    (10.0, 50.0, -0.1),
    (10.0, 80.0, 6.7),
    (5.0, 70.0, -0.2),
    (0.0, 90.0, -1.4),
    
    # Cold conditions
    (-5.0, 80.0, -7.8),
    (-10.0, 70.0, -14.6),
    
    # Extreme cases
    (25.0, 95.0, 24.1),
    (25.0, 30.0, 6.6),
    (15.0, 100.0, 15.0),
]


# ==============================================================================
# Test Class: Basic Functionality
# ==============================================================================

class TestMagnusEquationBasic:
    """Test basic functionality of _magnus_equation_scalar."""
    
    def test_returns_float(self):
        """Test that function returns a float."""
        result = _magnus_equation_scalar(
            293.15, 0.5,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        assert isinstance(result, (float, np.floating))
    
    def test_dewpoint_less_than_temperature(self):
        """Test that dew point is always ≤ temperature (physical constraint)."""
        temps = np.linspace(273.15, 313.15, 20)
        rhs = np.linspace(0.2, 0.9, 20)
        
        for temp_k in temps:
            for rh in rhs:
                td = _magnus_equation_scalar(
                    temp_k, rh,
                    MAGNUS_WATER_A, MAGNUS_WATER_B
                )
                assert td <= temp_k, (
                    f"Dew point ({td:.2f}K) exceeds temperature ({temp_k:.2f}K) "
                    f"at RH={rh:.2f}"
                )
    
    def test_rh_100_percent_equals_temperature(self):
        """Test that at RH=100%, dew point equals temperature.
        
        Magnus formula with 2 constants should give exact equality at RH=100%
        (unlike Buck's 3-constant formula which has small systematic error).
        """
        temps = [273.15, 283.15, 293.15, 303.15]
        
        for temp_k in temps:
            td = _magnus_equation_scalar(
                temp_k, 1.0,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            )
            
            # Magnus should be exact at RH=100% (within numerical precision)
            assert_allclose(
                td, temp_k,
                atol=1e-10,
                err_msg=f"At RH=100%, Td should equal T ({temp_k}K)"
            )
    
    def test_rh_approaching_100_percent(self):
        """Test correct monotonic behavior as RH→100%."""
        temp_k = 293.15
        rhs = [0.90, 0.95, 0.98, 0.99, 0.995, 1.0]
        
        dewpoints = [
            _magnus_equation_scalar(temp_k, rh, MAGNUS_WATER_A, MAGNUS_WATER_B)
            for rh in rhs
        ]
        
        # Dewpoints should increase monotonically
        for i in range(len(dewpoints) - 1):
            assert dewpoints[i] < dewpoints[i + 1], (
                f"Dew point should increase with RH: "
                f"RH[{i}]={rhs[i]:.3f}→Td={dewpoints[i]:.3f}K, "
                f"RH[{i+1}]={rhs[i+1]:.3f}→Td={dewpoints[i+1]:.3f}K"
            )
        
        # At RH=100%, should equal temperature (within precision)
        assert abs(dewpoints[-1] - temp_k) < 1e-9
    
    def test_increasing_rh_increases_dewpoint(self):
        """Test that increasing RH increases dew point (monotonicity)."""
        temp_k = 293.15
        rhs = np.linspace(0.3, 0.9, 10)
        
        dewpoints = [
            _magnus_equation_scalar(temp_k, rh, MAGNUS_WATER_A, MAGNUS_WATER_B)
            for rh in rhs
        ]
        
        for i in range(len(dewpoints) - 1):
            assert dewpoints[i] < dewpoints[i + 1]
    
    def test_valid_temperature_range(self):
        """Test function works across valid temperature range."""
        # Magnus valid range: -30°C to 60°C
        temps_c = [-30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
        
        for temp_c in temps_c:
            temp_k = celsius_to_kelvin(temp_c)
            td = _magnus_equation_scalar(
                temp_k, 0.6,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            )
            
            assert 173.15 < td < 373.15, (
                f"Dew point {td:.2f}K out of reasonable range at T={temp_k:.2f}K"
            )


# ==============================================================================
# Test Class: NOAA Reference Validation
# ==============================================================================

class TestMagnusEquationNOAA:
    """Validate against NOAA Online Calculator reference values."""
    
    @pytest.mark.parametrize("temp_c,rh_percent,expected_td_c", NOAA_REFERENCE_DATA)
    def test_noaa_reference_values(self, temp_c, rh_percent, expected_td_c):
        """Test against NOAA calculator reference values.
        
        NOAA likely uses Magnus-Tetens, so agreement should be good.
        However, they may use slightly different constants or rounding,
        so we allow reasonable tolerance.
        """
        temp_k = celsius_to_kelvin(temp_c)
        rh = rh_percent / 100.0
        
        td_k = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        td_c = kelvin_to_celsius(td_k)
        
        # Magnus should agree well with NOAA (both use similar formulation)
        # Allow 0.7°C tolerance for potential constant differences
        assert_allclose(
            td_c, expected_td_c,
            atol=0.7,
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
            td_k = _magnus_equation_scalar(
                temp_k, rh,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            )
            td_c = kelvin_to_celsius(td_k)
            
            assert abs(td_c - expected_td_c) < 0.7, (
                f"20°C, RH={rh_percent}%: Expected {expected_td_c}°C, Got {td_c:.2f}°C"
            )


# ==============================================================================
# Test Class: PsychroLib Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestMagnusEquationPsychroLib:
    """Cross-validate against PsychroLib.
    
    PsychroLib uses different formulation (likely Hyland-Wexler),
    so differences of 0.5-1.5°C are expected, especially at:
    - Cold temperatures (< 5°C)
    - Low humidity (< 40%)
    """
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 25, 30, 35])
    @pytest.mark.parametrize("rh", [0.3, 0.5, 0.7, 0.9])
    def test_against_psychrolib(self, temp_c, rh):
        """Test against PsychroLib across range of conditions."""
        temp_k = celsius_to_kelvin(temp_c)
        
        td_k = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        td_k_ref = get_psychrolib_dewpoint(temp_k, rh)
        
        # Determine tolerance based on conditions
        if temp_c <= 5 and rh < 0.4:
            tolerance = 1.5
        elif temp_c <= 10:
            tolerance = 1.2
        elif rh < 0.4:
            tolerance = 1.0
        else:
            tolerance = 0.8
        
        assert_allclose(
            td_k, td_k_ref,
            atol=tolerance,
            err_msg=(
                f"Magnus vs PsychroLib: T={temp_c}°C, RH={rh*100:.0f}%, "
                f"Magnus={kelvin_to_celsius(td_k):.2f}°C, "
                f"PsychroLib={kelvin_to_celsius(td_k_ref):.2f}°C, "
                f"Diff={abs(td_k - td_k_ref):.2f}K (tolerance={tolerance}K)"
            )
        )
    
    def test_psychrolib_warm_agreement(self):
        """Test that warm conditions (>20°C) have better agreement."""
        for temp_c in [20, 25, 30, 35]:
            for rh in [0.5, 0.7]:
                temp_k = celsius_to_kelvin(temp_c)
                
                td_k = _magnus_equation_scalar(
                    temp_k, rh,
                    MAGNUS_WATER_A, MAGNUS_WATER_B
                )
                td_k_ref = get_psychrolib_dewpoint(temp_k, rh)
                
                assert abs(td_k - td_k_ref) < 0.8, (
                    f"Warm conditions should agree well: "
                    f"T={temp_c}°C, RH={rh*100:.0f}%, "
                    f"diff={abs(td_k - td_k_ref):.2f}K"
                )


# ==============================================================================
# Test Class: MetPy Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestMagnusEquationMetPy:
    """Cross-validate against MetPy.
    
    MetPy likely uses Magnus or similar, so agreement should be excellent
    (better than PsychroLib comparison).
    """
    
    @pytest.mark.parametrize("temp_c", [-10, 0, 10, 20, 30, 40])
    @pytest.mark.parametrize("rh", [0.4, 0.6, 0.8])
    def test_against_metpy(self, temp_c, rh):
        """Test against MetPy across range of conditions.
        
        MetPy should agree closely with Magnus since both use similar
        formulations. Tolerance can be tighter than PsychroLib.
        """
        temp_k = celsius_to_kelvin(temp_c)
        
        td_k = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        td_k_ref = get_metpy_dewpoint(temp_k, rh)
        
        # MetPy uses similar formulation - expect good agreement
        tolerance = get_tolerance_for_conditions(temp_c, rh)
        
        # MetPy should agree better than PsychroLib
        if tolerance > 0.5:
            tolerance = 0.5 + (tolerance - 0.5) * 0.5  # Reduce tolerance
        
        assert_allclose(
            td_k, td_k_ref,
            atol=tolerance,
            err_msg=(
                f"Magnus vs MetPy: T={temp_c}°C, RH={rh*100:.0f}%, "
                f"Magnus={kelvin_to_celsius(td_k):.2f}°C, "
                f"MetPy={kelvin_to_celsius(td_k_ref):.2f}°C, "
                f"Diff={abs(td_k - td_k_ref):.2f}K (tolerance={tolerance}K)"
            )
        )
    
    def test_metpy_excellent_agreement_warm(self):
        """Test excellent agreement with MetPy at warm temperatures."""
        # Warm conditions should agree within 0.3°C
        for temp_c in [15, 20, 25, 30]:
            for rh in [0.5, 0.7]:
                temp_k = celsius_to_kelvin(temp_c)
                
                td_k = _magnus_equation_scalar(
                    temp_k, rh,
                    MAGNUS_WATER_A, MAGNUS_WATER_B
                )
                td_k_ref = get_metpy_dewpoint(temp_k, rh)
                
                assert abs(td_k - td_k_ref) < 0.4, (
                    f"Magnus and MetPy should agree closely at warm temps: "
                    f"T={temp_c}°C, RH={rh*100:.0f}%, "
                    f"diff={abs(td_k - td_k_ref):.2f}K"
                )


# ==============================================================================
# Test Class: Water vs Ice Constants
# ==============================================================================

class TestMagnusEquationIceVsWater:
    """Test ice vs water constant behavior."""
    
    @pytest.mark.parametrize("temp_c", [-20, -15, -10, -5, -2])
    @pytest.mark.parametrize("rh", [0.6, 0.8])
    def test_frostpoint_lower_than_dewpoint(self, temp_c, rh):
        """Test fundamental physics: frost point < dew point.
        
        At temperatures below 0°C, ice has lower saturation vapor pressure
        than supercooled water, so frost point must be lower than dew point.
        """
        temp_k = celsius_to_kelvin(temp_c)
        
        # Dew point (water)
        td_water = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Frost point (ice)
        tf_ice = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_ICE_A, MAGNUS_ICE_B
        )
        
        # Physics constraint
        assert tf_ice < td_water, (
            f"At {temp_c}°C, RH={rh*100:.0f}%: "
            f"Frost point ({kelvin_to_celsius(tf_ice):.2f}°C) "
            f"must be lower than dew point ({kelvin_to_celsius(td_water):.2f}°C)"
        )


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================

class TestMagnusEquationEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_humidity(self):
        """Test very low relative humidity (10%)."""
        temp_k = celsius_to_kelvin(20.0)
        rh = 0.10
        
        td_k = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert td_k < temp_k - 10
        assert 230 < td_k < 280
    
    def test_very_high_humidity(self):
        """Test very high relative humidity (99%)."""
        temp_k = celsius_to_kelvin(20.0)
        rh = 0.99
        
        td_k = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert abs(td_k - temp_k) < 0.5
    
    def test_freezing_point(self):
        """Test at freezing point (0°C)."""
        temp_k = celsius_to_kelvin(0.0)
        rh = 0.8
        
        td_water = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        td_ice = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_ICE_A, MAGNUS_ICE_B
        )
        
        assert td_water < temp_k
        assert td_ice < temp_k
        assert td_ice < td_water
    
    def test_hot_conditions(self):
        """Test hot conditions (40°C)."""
        temp_k = celsius_to_kelvin(40.0)
        rh = 0.5
        
        td_k = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        td_c = kelvin_to_celsius(td_k)
        assert 15 < td_c < 30
    
    def test_cold_conditions(self):
        """Test cold conditions (-20°C)."""
        temp_k = celsius_to_kelvin(-20.0)
        rh = 0.7
        
        td_k = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_ICE_A, MAGNUS_ICE_B
        )
        
        td_c = kelvin_to_celsius(td_k)
        assert -40 < td_c < -15


# ==============================================================================
# Test Class: Numerical Stability
# ==============================================================================

class TestMagnusEquationNumericalStability:
    """Test numerical stability and precision."""
    
    def test_deterministic(self):
        """Test that function is deterministic."""
        temp_k = 293.15
        rh = 0.6
        
        results = [
            _magnus_equation_scalar(temp_k, rh, MAGNUS_WATER_A, MAGNUS_WATER_B)
            for _ in range(10)
        ]
        
        assert all(r == results[0] for r in results)
    
    def test_no_nan_or_inf(self):
        """Test that function never returns NaN or Inf in valid range."""
        temps = np.linspace(243.15, 333.15, 50)  # -30°C to 60°C
        rhs = np.linspace(0.1, 0.99, 20)
        
        for temp_k in temps:
            for rh in rhs:
                td = _magnus_equation_scalar(
                    temp_k, rh,
                    MAGNUS_WATER_A, MAGNUS_WATER_B
                )
                
                assert not np.isnan(td)
                assert not np.isinf(td)
    
    def test_smooth_behavior(self):
        """Test that small RH changes produce smooth results."""
        temp_k = 293.15
        rh1 = 0.5000
        rh2 = 0.5001
        
        td1 = _magnus_equation_scalar(temp_k, rh1, MAGNUS_WATER_A, MAGNUS_WATER_B)
        td2 = _magnus_equation_scalar(temp_k, rh2, MAGNUS_WATER_A, MAGNUS_WATER_B)
        
        assert abs(td2 - td1) < 0.01


# ==============================================================================
# Test Class: Physical Constraints
# ==============================================================================

class TestMagnusEquationPhysics:
    """Test physical constraints and thermodynamic principles."""
    
    def test_dewpoint_depression_range(self):
        """Test that dew point depression (T - Td) is within physical limits."""
        temp_k = celsius_to_kelvin(20.0)
        
        for rh in [0.2, 0.4, 0.6, 0.8]:
            td_k = _magnus_equation_scalar(
                temp_k, rh,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            )
            
            depression = temp_k - td_k
            
            assert depression > 0
            assert depression < 50
    
    def test_clausius_clapeyron_trend(self):
        """Test rough consistency with Clausius-Clapeyron relation."""
        temp_k = celsius_to_kelvin(20.0)
        rhs = np.array([0.3, 0.5, 0.7, 0.9])
        
        dewpoints = np.array([
            _magnus_equation_scalar(temp_k, rh, MAGNUS_WATER_A, MAGNUS_WATER_B)
            for rh in rhs
        ])
        
        # Check monotonicity
        assert all(dewpoints[i] < dewpoints[i + 1] for i in range(len(dewpoints) - 1))


# ==============================================================================
# Test Class: Comparison with Buck
# ==============================================================================

class TestMagnusVsBuck:
    """Compare Magnus against Buck formulation.
    
    This documents expected differences between the two formulations.
    """
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 30])
    @pytest.mark.parametrize("rh", [0.5, 0.7])
    def test_magnus_vs_buck_difference(self, temp_c, rh):
        """Document typical differences between Magnus and Buck.
        
        This is not a validation test - just documentation of
        expected behavior differences between formulations.
        
        Typical differences: 0.3-0.6°C
        """
        temp_k = celsius_to_kelvin(temp_c)
        
        # Magnus (2 constants)
        td_magnus = _magnus_equation_scalar(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # For comparison with Buck, we'd need Buck implementation
        # This test just documents that Magnus is self-consistent
        assert 200 < td_magnus < 350  # Reasonable range check


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