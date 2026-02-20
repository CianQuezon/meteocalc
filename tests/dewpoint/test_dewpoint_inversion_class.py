"""
Comprehensive pytest suite for VaporInversionDewpoint class.

Tests exact dew point calculation using numerical solver with cross-validation
against NOAA, PsychroLib, and MetPy reference implementations.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import warnings

from meteocalc.dewpoint._dewpoint_equations import VaporInversionDewpoint
from meteocalc.vapor._enums import EquationName, SurfaceType

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
    """Get dew point from MetPy for validation."""
    if not HAS_METPY:
        pytest.skip("MetPy not available")
    
    temp_c = kelvin_to_celsius(temp_k)
    td_c = dewpoint_from_relative_humidity(
        temp_c * units.degC,
        rh * units.dimensionless
    ).magnitude
    return celsius_to_kelvin(td_c)


def get_psychrolib_dewpoint(temp_k, rh):
    """Get dew point from PsychroLib for validation."""
    if not HAS_PSYCHROLIB:
        pytest.skip("PsychroLib not available")
    
    temp_c = kelvin_to_celsius(temp_k)
    td_c = psy.GetTDewPointFromRelHum(temp_c, rh)
    return celsius_to_kelvin(td_c)


# ==============================================================================
# NOAA Reference Data
# ==============================================================================

# Reference values from NOAA dew point calculator
# Format: (temp_celsius, rh_fraction, expected_dewpoint_celsius)
NOAA_REFERENCE_DATA = [
    # Standard conditions
    (20.0, 0.50, 9.3),
    (20.0, 0.60, 12.0),
    (20.0, 0.70, 14.4),
    (20.0, 0.80, 16.4),
    (20.0, 0.90, 18.3),
    
    # Warm conditions
    (25.0, 0.50, 13.9),
    (30.0, 0.50, 18.4),
    (30.0, 0.70, 24.1),
    (35.0, 0.60, 26.1),
    
    # Cool conditions
    (15.0, 0.60, 7.6),
    (10.0, 0.50, -0.1),
    (10.0, 0.80, 6.7),
    (5.0, 0.70, -0.2),
]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def solver_water():
    """VaporInversionDewpoint solver for water surface."""
    return VaporInversionDewpoint(
        surface_type='water',
        vapor_equation_name='goff_gratch'
    )


@pytest.fixture
def solver_ice():
    """VaporInversionDewpoint solver for ice surface."""
    return VaporInversionDewpoint(
        surface_type='ice',
        vapor_equation_name='goff_gratch'
    )


@pytest.fixture(params=['goff_gratch', 'hyland_wexler'])
def solver_parametrized(request):
    """Parametrized solver across different equations."""
    return VaporInversionDewpoint(
        surface_type='water',
        vapor_equation_name=request.param
    )


# ==============================================================================
# Test Class: Initialization
# ==============================================================================

class TestVaporInversionInitialization:
    """Test proper initialization of VaporInversionDewpoint."""
    
    def test_init_with_string_enums(self):
        """Test initialization with string surface type and equation."""
        solver = VaporInversionDewpoint(
            surface_type='water',
            vapor_equation_name='goff_gratch'
        )
        
        assert solver.surface_type == SurfaceType.WATER
        assert solver.vapor_equation == EquationName.GOFF_GRATCH
    
    def test_init_with_enum_types(self):
        """Test initialization with enum types."""
        solver = VaporInversionDewpoint(
            surface_type=SurfaceType.WATER,
            vapor_equation_name=EquationName.GOFF_GRATCH
        )
        
        assert solver.surface_type == SurfaceType.WATER
        assert solver.vapor_equation == EquationName.GOFF_GRATCH
    
    def test_init_sets_temp_bounds(self):
        """Test that temperature bounds are set during initialization."""
        solver = VaporInversionDewpoint(
            surface_type='water',
            vapor_equation_name='goff_gratch'
        )
        
        assert hasattr(solver, 'temp_bounds')
        assert solver.temp_bounds is not None
    
    def test_init_ice_surface(self):
        """Test initialization with ice surface."""
        solver = VaporInversionDewpoint(
            surface_type='ice',
            vapor_equation_name='goff_gratch'
        )
        
        assert solver.surface_type == SurfaceType.ICE
    
    def test_init_different_equations(self):
        """Test initialization with different vapor equations."""
        equations = ['goff_gratch', 'hyland_wexler']
        
        for eq in equations:
            solver = VaporInversionDewpoint(
                surface_type='water',
                vapor_equation_name=eq
            )
            assert solver.vapor_equation == EquationName[eq.upper()]


# ==============================================================================
# Test Class: Scalar Calculations
# ==============================================================================

class TestVaporInversionScalarCalculations:
    """Test scalar input/output behavior."""
    
    def test_scalar_input_returns_scalar(self, solver_water):
        """Test that scalar inputs return scalar output."""
        temp_k = 293.15
        rh = 0.6
        
        result = solver_water.calculate(temp_k=temp_k, rh=rh)
        
        assert isinstance(result, (float, np.floating))
        assert not isinstance(result, np.ndarray)
    
    def test_scalar_value_physically_reasonable(self, solver_water):
        """Test that scalar result is physically reasonable."""
        temp_k = 293.15  # 20°C
        rh = 0.6
        
        td = solver_water.calculate(temp_k=temp_k, rh=rh)
        td_c = kelvin_to_celsius(td)
        
        # At 20°C, 60% RH, dew point should be ~12°C
        assert 10 < td_c < 14
        assert td < temp_k  # Dew point must be less than temperature
    
    def test_scalar_high_humidity(self, solver_water):
        """Test scalar calculation at high humidity."""
        temp_k = 293.15
        rh = 0.95
        
        td = solver_water.calculate(temp_k=temp_k, rh=rh)
        
        # At 95% RH, dew point should be very close to temperature
        assert abs(td - temp_k) < 2.0  # Within 2K
    
    def test_scalar_low_humidity(self, solver_water):
        """Test scalar calculation at low humidity."""
        temp_k = 293.15
        rh = 0.2
        
        td = solver_water.calculate(temp_k=temp_k, rh=rh)
        
        # At 20% RH, large depression expected
        assert temp_k - td > 10  # More than 10K depression


# ==============================================================================
# Test Class: Vector Calculations
# ==============================================================================

class TestVaporInversionVectorCalculations:
    """Test vector/array input behavior."""
    
    def test_vector_input_returns_array(self, solver_water):
        """Test that array inputs return array output."""
        temps = np.array([273.15, 283.15, 293.15])
        rhs = np.array([0.5, 0.6, 0.7])
        
        result = solver_water.calculate(temp_k=temps, rh=rhs)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(temps)
    
    def test_vector_preserves_shape(self, solver_water):
        """Test that output shape matches input shape."""
        n = 100
        temps = np.linspace(273.15, 313.15, n)
        rhs = np.full(n, 0.6)
        
        result = solver_water.calculate(temp_k=temps, rh=rhs)
        
        assert result.shape == (n,)
    
    def test_vector_monotonicity_with_temperature(self, solver_water):
        """Test that dew point increases with temperature at constant RH."""
        temps = np.linspace(273.15, 313.15, 20)
        rh = 0.6
        
        dewpoints = solver_water.calculate(temp_k=temps, rh=rh)
        
        # Check monotonicity
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0), "Dew point should increase with temperature"
    
    def test_vector_monotonicity_with_humidity(self, solver_water):
        """Test that dew point increases with RH at constant temperature."""
        temp = 293.15
        rhs = np.linspace(0.3, 0.9, 20)
        
        dewpoints = solver_water.calculate(temp_k=temp, rh=rhs)
        
        # Check monotonicity
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0), "Dew point should increase with RH"
    
    def test_vector_broadcast_scalar_rh(self, solver_water):
        """Test broadcasting scalar RH with vector temperatures."""
        temps = np.array([273.15, 283.15, 293.15])
        rh = 0.6  # Scalar
        
        result = solver_water.calculate(temp_k=temps, rh=rh)
        
        assert len(result) == len(temps)
        assert isinstance(result, np.ndarray)


# ==============================================================================
# Test Class: NOAA Validation
# ==============================================================================

class TestVaporInversionNOAAValidation:
    """Validate against NOAA dew point calculator reference values."""
    
    @pytest.mark.parametrize("temp_c,rh,expected_td_c", NOAA_REFERENCE_DATA)
    def test_noaa_reference_values(self, solver_water, temp_c, rh, expected_td_c):
        """Test against NOAA calculator reference values.
        
        Exact numerical solver should match NOAA within ±0.3°C.
        """
        temp_k = celsius_to_kelvin(temp_c)
        
        td_k = solver_water.calculate(temp_k=temp_k, rh=rh)
        td_c = kelvin_to_celsius(td_k)
        
        assert_allclose(
            td_c, expected_td_c,
            atol=0.3,
            err_msg=(
                f"Solver vs NOAA at T={temp_c}°C, RH={rh*100:.0f}%: "
                f"Expected={expected_td_c}°C, Got={td_c:.2f}°C"
            )
        )
    
    def test_noaa_standard_condition_accuracy(self, solver_water):
        """Test exceptional accuracy at standard conditions (20°C, 60% RH)."""
        temp_k = celsius_to_kelvin(20.0)
        rh = 0.6
        expected_td_c = 12.0
        
        td_k = solver_water.calculate(temp_k=temp_k, rh=rh)
        td_c = kelvin_to_celsius(td_k)
        
        # Should be very accurate at standard conditions
        assert abs(td_c - expected_td_c) < 0.1, \
            f"At standard conditions, error should be < 0.1°C, got {abs(td_c - expected_td_c):.4f}°C"


# ==============================================================================
# Test Class: MetPy Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestVaporInversionMetPyValidation:
    """Cross-validate against MetPy implementation."""
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 25, 30])
    @pytest.mark.parametrize("rh", [0.4, 0.6, 0.8])
    def test_metpy_agreement(self, solver_water, temp_c, rh):
        """Test agreement with MetPy across temperature and humidity range."""
        temp_k = celsius_to_kelvin(temp_c)
        
        td_solver = solver_water.calculate(temp_k=temp_k, rh=rh)
        td_metpy = get_metpy_dewpoint(temp_k, rh)
        
        # Should agree within 0.5°C
        diff = abs(td_solver - td_metpy)
        assert diff < 0.5, \
            f"Solver vs MetPy at T={temp_c}°C, RH={rh*100:.0f}%: diff={diff:.4f}K"
    
    def test_metpy_excellent_agreement_standard_conditions(self, solver_water):
        """Test very close agreement with MetPy at standard conditions."""
        # Standard meteorological conditions
        test_cases = [
            (20, 0.5),
            (25, 0.6),
            (30, 0.7),
        ]
        
        for temp_c, rh in test_cases:
            temp_k = celsius_to_kelvin(temp_c)
            
            td_solver = solver_water.calculate(temp_k=temp_k, rh=rh)
            td_metpy = get_metpy_dewpoint(temp_k, rh)
            
            diff = abs(td_solver - td_metpy)
            assert diff < 0.2, \
                f"Poor agreement at T={temp_c}°C, RH={rh*100:.0f}%: diff={diff:.4f}K"


# ==============================================================================
# Test Class: PsychroLib Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestVaporInversionPsychroLibValidation:
    """Cross-validate against PsychroLib (ASHRAE standard)."""
    
    def _get_tolerance(self, temp_c, rh):
        """Get appropriate tolerance based on conditions."""
        if temp_c <= 0 and rh < 0.35:
            return 2.0
        elif temp_c <= 0 and rh < 0.6:
            return 1.2
        elif temp_c <= 5:
            return 1.0
        elif temp_c <= 15 and rh < 0.4:
            return 1.0
        elif rh < 0.4:
            return 0.8
        else:
            return 0.5
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 25, 30, 35])
    @pytest.mark.parametrize("rh", [0.3, 0.5, 0.7, 0.9])
    def test_psychrolib_agreement(self, temp_c, rh):
        """Test agreement with PsychroLib across conditions."""
        # Use Hyland-Wexler to match PsychroLib
        solver = VaporInversionDewpoint(
            surface_type='water',
            vapor_equation_name='hyland_wexler'
        )
        
        temp_k = celsius_to_kelvin(temp_c)
        
        td_solver = solver.calculate(temp_k=temp_k, rh=rh)
        td_psychrolib = get_psychrolib_dewpoint(temp_k, rh)
        
        tolerance = self._get_tolerance(temp_c, rh)
        diff = abs(td_solver - td_psychrolib)
        
        assert diff <= tolerance, \
            f"Solver vs PsychroLib at T={temp_c}°C, RH={rh*100:.0f}%: " \
            f"diff={diff:.4f}K (tolerance={tolerance}K)"
    
    def test_psychrolib_standard_conditions_excellent(self):
        """Test excellent agreement at standard conditions."""
        solver = VaporInversionDewpoint(
            surface_type='water',
            vapor_equation_name='hyland_wexler'
        )
        
        test_cases = [
            (20, 0.5),
            (25, 0.6),
            (30, 0.7),
        ]
        
        for temp_c, rh in test_cases:
            temp_k = celsius_to_kelvin(temp_c)
            
            td_solver = solver.calculate(temp_k=temp_k, rh=rh)
            td_psychrolib = get_psychrolib_dewpoint(temp_k, rh)
            
            diff = abs(td_solver - td_psychrolib)
            assert diff < 0.3, \
                f"Standard conditions should agree within 0.3K, got {diff:.4f}K"


# ==============================================================================
# Test Class: Physical Correctness
# ==============================================================================

class TestVaporInversionPhysicalCorrectness:
    """Test that results satisfy physical constraints."""
    
    def test_dewpoint_less_than_temperature_always(self, solver_water):
        """Test that Td ≤ T always (fundamental constraint)."""
        temps = np.linspace(273.15, 313.15, 50)
        rhs = np.linspace(0.2, 0.98, 50)
        
        dewpoints = solver_water.calculate(temp_k=temps, rh=rhs)
        
        assert_array_less(
            dewpoints, temps + 0.01,  # Small tolerance for numerical precision
            err_msg="Dew point exceeds temperature (physics violation!)"
        )
    
    def test_dewpoint_equals_temperature_at_saturation(self, solver_water):
        """Test that Td ≈ T at RH=100%."""
        temps = np.array([273.15, 293.15, 313.15])
        rh = 1.0
        
        dewpoints = solver_water.calculate(temp_k=temps, rh=rh)
        
        # At saturation, dew point should equal temperature
        assert_allclose(
            dewpoints, temps,
            atol=0.1,
            err_msg="At RH=100%, Td should equal T"
        )
    
    def test_frost_point_greater_than_dewpoint_below_freezing(self):
        """Test that frost point > dew point below 0°C."""
        temp_k = np.array([263.15, 268.15])  # -10°C, -5°C
        rh = np.array([0.7, 0.7])
        
        # Dew point (water)
        solver_water = VaporInversionDewpoint(
            surface_type='water',
            vapor_equation_name='goff_gratch'
        )
        td_water = solver_water.calculate(temp_k=temp_k, rh=rh)
        
        # Frost point (ice)
        solver_ice = VaporInversionDewpoint(
            surface_type='ice',
            vapor_equation_name='goff_gratch'
        )
        tf_ice = solver_ice.calculate(temp_k=temp_k, rh=rh)
        
        # Frost point should be higher than dew point
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.all(tf_ice > td_water), \
                "Frost point should be > dew point below 0°C"
    
    def test_no_nan_or_inf_values(self, solver_water):
        """Test that results never contain NaN or Inf."""
        temps = np.linspace(273.15, 313.15, 100)
        rhs = np.linspace(0.1, 0.99, 100)
        
        dewpoints = solver_water.calculate(temp_k=temps, rh=rhs)
        
        assert not np.any(np.isnan(dewpoints)), "Results contain NaN"
        assert not np.any(np.isinf(dewpoints)), "Results contain Inf"
    
    def test_dewpoint_depression_reasonable_range(self, solver_water):
        """Test that dew point depression is in reasonable range."""
        temp_k = 293.15  # 20°C
        rhs = np.array([0.2, 0.5, 0.8])
        
        dewpoints = solver_water.calculate(temp_k=temp_k, rh=rhs)
        depressions = temp_k - dewpoints
        
        # Depression should be positive and less than 50K
        assert np.all(depressions > 0), "Negative dew point depression"
        assert np.all(depressions < 50), "Unreasonably large dew point depression"


# ==============================================================================
# Test Class: Multiple Vapor Equations
# ==============================================================================

class TestVaporInversionMultipleEquations:
    """Test behavior with different vapor pressure equations."""
    
    def test_goff_gratch_equation(self):
        """Test with Goff-Gratch equation (WMO standard)."""
        solver = VaporInversionDewpoint(
            surface_type='water',
            vapor_equation_name='goff_gratch'
        )
        
        td = solver.calculate(temp_k=293.15, rh=0.6)
        td_c = kelvin_to_celsius(td)
        
        assert 11 < td_c < 13  # Should be ~12°C
    
    def test_hyland_wexler_equation(self):
        """Test with Hyland-Wexler equation (ASHRAE standard)."""
        solver = VaporInversionDewpoint(
            surface_type='water',
            vapor_equation_name='hyland_wexler'
        )
        
        td = solver.calculate(temp_k=293.15, rh=0.6)
        td_c = kelvin_to_celsius(td)
        
        assert 11 < td_c < 13  # Should be ~12°C
    
    def test_equations_agree_at_standard_conditions(self):
        """Test that different equations agree at standard conditions."""
        temp_k = 293.15
        rh = 0.6
        
        equations = ['goff_gratch', 'hyland_wexler']
        results = []
        
        for eq in equations:
            solver = VaporInversionDewpoint(
                surface_type='water',
                vapor_equation_name=eq
            )
            td = solver.calculate(temp_k=temp_k, rh=rh)
            results.append(td)
        
        # All equations should agree within 0.1K at standard conditions
        max_diff = max(results) - min(results)
        assert max_diff < 0.1, \
            f"Equations disagree by {max_diff:.4f}K (should be < 0.1K)"


# ==============================================================================
# Test Class: Edge Cases and Limits
# ==============================================================================

class TestVaporInversionEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_humidity(self, solver_water):
        """Test at very low humidity (10%)."""
        temp_k = 293.15
        rh = 0.1
        
        td = solver_water.calculate(temp_k=temp_k, rh=rh)
        td_c = kelvin_to_celsius(td)
        
        # At 20°C, 10% RH → large depression
        assert td_c < 0  # Should be below freezing
        assert td_c > -30  # But not unreasonably low
    
    def test_very_high_humidity(self, solver_water):
        """Test at very high humidity (99%)."""
        temp_k = 293.15
        rh = 0.99
        
        td = solver_water.calculate(temp_k=temp_k, rh=rh)
        
        # At 99% RH, dew point very close to temperature
        assert abs(td - temp_k) < 0.5
    
    def test_cold_temperature(self, solver_water):
        """Test at cold temperature (0°C)."""
        temp_k = 273.15
        rh = 0.6
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            td = solver_water.calculate(temp_k=temp_k, rh=rh)
        
        assert td < temp_k
        assert td > 250  # Physically reasonable
    
    def test_hot_temperature(self, solver_water):
        """Test at hot temperature (40°C)."""
        temp_k = 313.15
        rh = 0.5
        
        td = solver_water.calculate(temp_k=temp_k, rh=rh)
        td_c = kelvin_to_celsius(td)
        
        assert 25 < td_c < 30  # Reasonable range
    
    def test_empty_array(self, solver_water):
        """Test with empty arrays."""
        temps = np.array([])
        rhs = np.array([])
        
        # Should handle empty arrays gracefully
        try:
            result = solver_water.calculate(temp_k=temps, rh=rhs)
            assert len(result) == 0
        except (ValueError, IndexError):
            # Acceptable to raise error for empty input
            pass


# ==============================================================================
# Test Class: Performance
# ==============================================================================

class TestVaporInversionPerformance:
    """Test performance characteristics."""
    
    def test_handles_large_arrays(self, solver_water):
        """Test that solver can handle large arrays."""
        n = 10000
        temps = np.linspace(273.15, 313.15, n)
        rhs = np.full(n, 0.6)
        
        dewpoints = solver_water.calculate(temp_k=temps, rh=rhs)
        
        assert len(dewpoints) == n
        assert isinstance(dewpoints, np.ndarray)
        assert not np.any(np.isnan(dewpoints))
    
    @pytest.mark.slow
    def test_performance_acceptable(self, solver_water):
        """Test that performance meets minimum requirements."""
        import time
        
        n = 10000
        temps = np.linspace(273.15, 313.15, n)
        rhs = np.full(n, 0.6)
        
        # Warmup
        _ = solver_water.calculate(
            temp_k=np.array([293.15]),
            rh=np.array([0.6])
        )
        
        # Benchmark
        start = time.perf_counter()
        dewpoints = solver_water.calculate(temp_k=temps, rh=rhs)
        elapsed = time.perf_counter() - start
        
        calc_per_sec = n / elapsed
        
        # Should achieve at least 1000 calc/sec
        assert calc_per_sec > 1000, \
            f"Performance too low: {calc_per_sec:.0f} calc/sec (need >1000)"


# ==============================================================================
# Test Class: Error Handling
# ==============================================================================

class TestVaporInversionErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_rh_raises_error(self, solver_water):
        """Test that RH outside [0, 1] raises error."""
        temp_k = 293.15
        
        with pytest.raises((ValueError, AssertionError)):
            solver_water.calculate(temp_k=temp_k, rh=1.5)
        
        with pytest.raises((ValueError, AssertionError)):
            solver_water.calculate(temp_k=temp_k, rh=-0.1)
    
    def test_mismatched_array_shapes(self, solver_water):
        """Test that mismatched array shapes are handled."""
        temps = np.array([273.15, 283.15, 293.15])
        rhs = np.array([0.5, 0.6])  # Different length
        
        # Should either broadcast or raise error
        try:
            result = solver_water.calculate(temp_k=temps, rh=rhs)
            # If it succeeds, check it makes sense
            assert len(result) in [len(temps), len(rhs)]
        except ValueError:
            # Acceptable to raise error for mismatched shapes
            pass


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