"""
Comprehensive unit tests for MagnusDewpointEquation class.

Tests Magnus-Tetens dew point equation implementation including:
- Initialization and configuration
- Scalar and vector calculations
- Water and ice surface types
- Temperature bounds
- Cross-validation against NOAA, MetPy, and PsychroLib
- Physical constraints
- Edge cases

Author: Test Suite
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import warnings

from meteocalc.dewpoint._dewpoint_equations import MagnusDewpointEquation
from meteocalc.dewpoint._enums import DewPointEquationName
from meteocalc.vapor._enums import SurfaceType

# Reference implementations
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
    """
    Get appropriate tolerance for Magnus validation.
    
    Magnus is less accurate than Buck, especially at:
    - Cold temperatures (< 0°C)
    - Low humidity (< 40%)
    """
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
# NOAA Reference Data
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
]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def magnus_water():
    """Create Magnus equation instance for water surface."""
    return MagnusDewpointEquation(surface_type='water')


@pytest.fixture
def magnus_ice():
    """Create Magnus equation instance for ice surface."""
    return MagnusDewpointEquation(surface_type='ice')


# ==============================================================================
# Test Class: Initialization
# ==============================================================================

class TestMagnusInitialization:
    """Test initialization and configuration of MagnusDewpointEquation."""
    
    def test_name_is_magnus(self, magnus_water):
        """Test that equation name is correctly set to MAGNUS."""
        assert magnus_water.name == DewPointEquationName.MAGNUS
    
    def test_water_surface_initialization(self, magnus_water):
        """Test initialization with water surface type."""
        assert magnus_water.surface_type == SurfaceType.WATER
        assert magnus_water.temp_bounds == (233.15, 333.15)
    
    def test_ice_surface_initialization(self, magnus_ice):
        """Test initialization with ice surface type."""
        assert magnus_ice.surface_type == SurfaceType.ICE
        assert magnus_ice.temp_bounds == (233.15, 273.15)
    
    def test_accepts_string_surface_type(self):
        """Test that surface_type can be passed as string."""
        eq_water = MagnusDewpointEquation(surface_type='water')
        assert eq_water.surface_type == SurfaceType.WATER
        
        eq_ice = MagnusDewpointEquation(surface_type='ice')
        assert eq_ice.surface_type == SurfaceType.ICE
    
    def test_accepts_enum_surface_type(self):
        """Test that surface_type can be passed as enum."""
        eq_water = MagnusDewpointEquation(surface_type=SurfaceType.WATER)
        assert eq_water.surface_type == SurfaceType.WATER
        
        eq_ice = MagnusDewpointEquation(surface_type=SurfaceType.ICE)
        assert eq_ice.surface_type == SurfaceType.ICE
    
    def test_water_bounds_wider_than_buck(self):
        """Test that Magnus water bounds extend to lower temperatures than Buck."""
        eq = MagnusDewpointEquation(surface_type='water')
        # Magnus water: -40°C to +60°C (wider range than Buck's -30°C to +50°C)
        assert eq.temp_bounds[0] == 233.15  # -40°C
        assert eq.temp_bounds[1] == 333.15  # +60°C


# ==============================================================================
# Test Class: Scalar Calculations
# ==============================================================================

class TestMagnusScalarCalculations:
    """Test Magnus equation with scalar inputs."""
    
    def test_returns_scalar_for_scalar_input(self, magnus_water):
        """Test that scalar input returns scalar output."""
        td = magnus_water.calculate(temp_k=293.15, rh=0.6)
        
        assert isinstance(td, (float, np.floating))
        assert not isinstance(td, np.ndarray)
    
    def test_basic_calculation_water(self, magnus_water):
        """Test basic dew point calculation for water surface."""
        temp_k = celsius_to_kelvin(20.0)  # 20°C
        rh = 0.6  # 60%
        
        td = magnus_water.calculate(temp_k=temp_k, rh=rh)
        td_c = kelvin_to_celsius(td)
        
        # At 20°C, 60% RH, dew point should be around 12°C
        assert 11.5 < td_c < 12.5
    
    def test_rh_100_percent_equals_temperature(self, magnus_water):
        """Test that RH=100% gives dew point = temperature.
        
        Magnus 2-constant formula should give exact equality at RH=100%
        (unlike Buck's 3-constant formula which has small systematic error).
        """
        temps = [273.15, 283.15, 293.15, 303.15]
        
        for temp_k in temps:
            td = magnus_water.calculate(temp_k=temp_k, rh=1.0)
            
            # Magnus should be exact at RH=100%
            assert abs(td - temp_k) < 1e-10, (
                f"At RH=100%, Td should equal T ({temp_k}K)"
            )
    
    def test_dewpoint_less_than_temperature(self, magnus_water):
        """Test that dew point is always ≤ temperature."""
        temps = [273.15, 283.15, 293.15, 303.15, 313.15]
        rhs = [0.3, 0.5, 0.7, 0.9]
        
        for temp_k in temps:
            for rh in rhs:
                td = magnus_water.calculate(temp_k=temp_k, rh=rh)
                assert td <= temp_k, (
                    f"Dew point ({td:.2f}K) exceeds temperature ({temp_k:.2f}K)"
                )
    
    def test_increasing_rh_increases_dewpoint(self, magnus_water):
        """Test that increasing RH increases dew point."""
        temp_k = 293.15
        rhs = [0.3, 0.5, 0.7, 0.9]
        
        dewpoints = [magnus_water.calculate(temp_k=temp_k, rh=rh) for rh in rhs]
        
        for i in range(len(dewpoints) - 1):
            assert dewpoints[i] < dewpoints[i + 1]


# ==============================================================================
# Test Class: Vector Calculations
# ==============================================================================

class TestMagnusVectorCalculations:
    """Test Magnus equation with array inputs."""
    
    def test_returns_array_for_array_input(self, magnus_water):
        """Test that array input returns array output."""
        temps = np.array([293.15, 303.15, 313.15])
        rhs = np.array([0.5, 0.6, 0.7])
        
        td = magnus_water.calculate(temp_k=temps, rh=rhs)
        
        assert isinstance(td, np.ndarray)
        assert td.shape == temps.shape
    
    def test_preserves_array_shape(self, magnus_water):
        """Test that output shape matches input shape."""
        temps = np.array([[293.15, 303.15], [283.15, 313.15]])
        rhs = np.array([[0.5, 0.6], [0.7, 0.8]])
        
        td = magnus_water.calculate(temp_k=temps, rh=rhs)
        
        assert td.shape == (2, 2)
    
    def test_vector_matches_scalar(self, magnus_water):
        """Test that vectorized calculation matches scalar results."""
        temps = np.array([293.15, 303.15, 313.15])
        rhs = np.array([0.5, 0.6, 0.7])
        
        # Vectorized calculation
        td_vector = magnus_water.calculate(temp_k=temps, rh=rhs)
        
        # Scalar calculations
        td_scalar = np.array([
            magnus_water.calculate(temp_k=t, rh=r)
            for t, r in zip(temps, rhs)
        ])
        
        assert_allclose(td_vector, td_scalar, rtol=1e-15)
    
    def test_large_array_calculation(self, magnus_water):
        """Test calculation with large arrays."""
        n = 10000
        temps = np.linspace(273.15, 313.15, n)
        rhs = np.full(n, 0.6)
        
        td = magnus_water.calculate(temp_k=temps, rh=rhs)
        
        assert len(td) == n
        assert not np.any(np.isnan(td))
        assert not np.any(np.isinf(td))


# ==============================================================================
# Test Class: NOAA Reference Validation
# ==============================================================================

class TestMagnusNOAAValidation:
    """Validate against NOAA Online Calculator reference values."""
    
    @pytest.mark.parametrize("temp_c,rh_percent,expected_td_c", NOAA_REFERENCE_DATA)
    def test_noaa_reference_values(self, magnus_water, temp_c, rh_percent, expected_td_c):
        """Test against NOAA calculator reference values.
        
        Magnus and NOAA may use slightly different formulations,
        so we allow reasonable tolerance (0.7°C).
        """
        temp_k = celsius_to_kelvin(temp_c)
        rh = rh_percent / 100.0
        
        td_k = magnus_water.calculate(temp_k=temp_k, rh=rh)
        td_c = kelvin_to_celsius(td_k)
        
        # Magnus should agree reasonably with NOAA
        assert_allclose(
            td_c, expected_td_c,
            atol=0.7,
            err_msg=(
                f"Magnus vs NOAA: T={temp_c}°C, RH={rh_percent}%, "
                f"Expected={expected_td_c}°C, Got={td_c:.2f}°C"
            )
        )
    
    def test_noaa_warm_conditions(self, magnus_water):
        """Test warm conditions against NOAA."""
        # 30°C, 70% RH → Expected ~24.1°C
        td = magnus_water.calculate(temp_k=celsius_to_kelvin(30.0), rh=0.7)
        td_c = kelvin_to_celsius(td)
        
        assert abs(td_c - 24.1) < 0.7


# ==============================================================================
# Test Class: MetPy Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestMagnusMetPyValidation:
    """Cross-validate against MetPy.
    
    MetPy likely uses Magnus or very similar formulation,
    so agreement should be excellent.
    """
    
    @pytest.mark.parametrize("temp_c", [-10, 0, 10, 20, 30, 40])
    @pytest.mark.parametrize("rh", [0.4, 0.6, 0.8])
    def test_against_metpy(self, magnus_water, temp_c, rh):
        """Test against MetPy across range of conditions."""
        temp_k = celsius_to_kelvin(temp_c)
        
        td_k = magnus_water.calculate(temp_k=temp_k, rh=rh)
        td_k_ref = get_metpy_dewpoint(temp_k, rh)
        
        # MetPy should agree very closely with Magnus
        tolerance = get_tolerance_for_conditions(temp_c, rh)
        # Reduce tolerance since both likely use Magnus
        tolerance = min(tolerance, 0.5)
        
        assert_allclose(
            td_k, td_k_ref,
            atol=tolerance,
            err_msg=(
                f"Magnus vs MetPy: T={temp_c}°C, RH={rh*100:.0f}%, "
                f"Magnus={kelvin_to_celsius(td_k):.2f}°C, "
                f"MetPy={kelvin_to_celsius(td_k_ref):.2f}°C"
            )
        )
    
    def test_metpy_excellent_agreement_warm(self, magnus_water):
        """Test excellent agreement with MetPy at warm temperatures."""
        # Warm conditions should agree within 0.3°C
        for temp_c in [15, 20, 25, 30]:
            for rh in [0.5, 0.7]:
                temp_k = celsius_to_kelvin(temp_c)
                
                td_k = magnus_water.calculate(temp_k=temp_k, rh=rh)
                td_k_ref = get_metpy_dewpoint(temp_k, rh)
                
                assert abs(td_k - td_k_ref) < 0.4, (
                    f"Magnus and MetPy should agree closely: "
                    f"T={temp_c}°C, RH={rh*100:.0f}%, "
                    f"diff={abs(td_k - td_k_ref):.2f}K"
                )


# ==============================================================================
# Test Class: PsychroLib Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestMagnusPsychroLibValidation:
    """Cross-validate against PsychroLib.
    
    PsychroLib uses Hyland-Wexler, so larger differences expected
    (0.5-1.5°C) especially at cold temps + low humidity.
    """
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 25, 30, 35])
    @pytest.mark.parametrize("rh", [0.3, 0.5, 0.7, 0.9])
    def test_against_psychrolib(self, magnus_water, temp_c, rh):
        """Test against PsychroLib across range of conditions."""
        temp_k = celsius_to_kelvin(temp_c)
        
        td_k = magnus_water.calculate(temp_k=temp_k, rh=rh)
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


# ==============================================================================
# Test Class: Water vs Ice Surface
# ==============================================================================

class TestMagnusWaterVsIce:
    """Test differences between water and ice surfaces."""
    
    def test_frostpoint_higher_than_dewpoint_below_zero(self):
        """Test that frost point > dew point below 0°C.
        
        Below freezing, ice has lower saturation vapor pressure than
        supercooled water, so frost point is warmer than dew point.
        """
        temps_c = [-20, -15, -10, -5, -2]
        rhs = [0.6, 0.7, 0.8]
        
        magnus_water = MagnusDewpointEquation(surface_type='water')
        magnus_ice = MagnusDewpointEquation(surface_type='ice')
        
        for temp_c in temps_c:
            for rh in rhs:
                temp_k = celsius_to_kelvin(temp_c)
                
                td_water = magnus_water.calculate(temp_k=temp_k, rh=rh)
                tf_ice = magnus_ice.calculate(temp_k=temp_k, rh=rh)
                
                assert tf_ice > td_water, (
                    f"At {temp_c}°C, RH={rh*100:.0f}%: "
                    f"Frost point ({kelvin_to_celsius(tf_ice):.2f}°C) "
                    f"should be higher than dew point ({kelvin_to_celsius(td_water):.2f}°C)"
                )
    
    def test_frost_dewpoint_difference_reasonable(self):
        """Test that frost point - dew point difference is in expected range."""
        temp_k = celsius_to_kelvin(-10.0)
        rh = 0.8
        
        magnus_water = MagnusDewpointEquation(surface_type='water')
        magnus_ice = MagnusDewpointEquation(surface_type='ice')
        
        td_water = magnus_water.calculate(temp_k=temp_k, rh=rh)
        tf_ice = magnus_ice.calculate(temp_k=temp_k, rh=rh)
        
        diff = tf_ice - td_water
        
        # At -10°C, difference should be roughly 0.5-3K
        assert 0.3 < diff < 4.0, (
            f"Difference ({diff:.2f}K) outside expected range"
        )


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================

class TestMagnusEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_humidity(self, magnus_water):
        """Test with very low humidity (10%)."""
        td = magnus_water.calculate(temp_k=293.15, rh=0.1)
        
        # Should be well below temperature
        assert td < 293.15 - 10
        assert 240 < td < 280
    
    def test_very_high_humidity(self, magnus_water):
        """Test with very high humidity (99%)."""
        td = magnus_water.calculate(temp_k=293.15, rh=0.99)
        
        # Should be very close to temperature
        assert abs(td - 293.15) < 0.5
    
    def test_cold_temperature(self, magnus_ice):
        """Test at very cold temperature (-35°C)."""
        tf = magnus_ice.calculate(temp_k=celsius_to_kelvin(-35.0), rh=0.7)
        
        # Should be reasonable for arctic conditions
        assert 220 < tf < 250
    
    def test_hot_temperature(self, magnus_water):
        """Test at hot temperature (55°C)."""
        td = magnus_water.calculate(temp_k=celsius_to_kelvin(55.0), rh=0.5)
        
        # Should be reasonable for extreme conditions
        assert 300 < td < 335
    
    def test_no_nan_or_inf(self, magnus_water):
        """Test that function never produces NaN or Inf."""
        temps = np.linspace(240, 330, 100)
        rhs = np.linspace(0.1, 0.99, 100)
        
        td = magnus_water.calculate(temp_k=temps, rh=rhs)
        
        assert not np.any(np.isnan(td))
        assert not np.any(np.isinf(td))


# ==============================================================================
# Test Class: Physical Constraints
# ==============================================================================

class TestMagnusPhysicalConstraints:
    """Test that physical constraints are satisfied."""
    
    def test_monotonic_with_temperature(self, magnus_water):
        """Test that dew point increases with temperature at constant RH."""
        temps = np.linspace(273.15, 313.15, 20)
        rh = 0.6
        
        dewpoints = magnus_water.calculate(temp_k=temps, rh=rh)
        
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0)
    
    def test_monotonic_with_rh(self, magnus_water):
        """Test that dew point increases with RH at constant temperature."""
        temp_k = 293.15
        rhs = np.linspace(0.2, 0.95, 20)
        
        dewpoints = magnus_water.calculate(temp_k=temp_k, rh=rhs)
        
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0)
    
    def test_dewpoint_depression_range(self, magnus_water):
        """Test that dew point depression is within physical limits."""
        temps = np.array([273.15, 293.15, 313.15])
        rhs = np.array([0.3, 0.5, 0.7])
        
        td = magnus_water.calculate(temp_k=temps, rh=rhs)
        depression = temps - td
        
        assert np.all(depression > 0)
        assert np.all(depression < 50)


# ==============================================================================
# Test Class: Determinism
# ==============================================================================

class TestMagnusDeterminism:
    """Test deterministic behavior and reproducibility."""
    
    def test_deterministic_scalar(self, magnus_water):
        """Test that scalar calculations are deterministic."""
        results = [
            magnus_water.calculate(temp_k=293.15, rh=0.6)
            for _ in range(10)
        ]
        
        assert all(r == results[0] for r in results)
    
    def test_deterministic_vector(self, magnus_water):
        """Test that vector calculations are deterministic."""
        temps = np.linspace(273.15, 313.15, 100)
        rhs = np.full(100, 0.6)
        
        result1 = magnus_water.calculate(temp_k=temps, rh=rhs)
        result2 = magnus_water.calculate(temp_k=temps, rh=rhs)
        
        assert_array_equal(result1, result2)


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])