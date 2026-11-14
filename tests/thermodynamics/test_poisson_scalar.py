"""
Comprehensive test suite for _poisson_scalar()

Tests Poisson's equation (potential temperature calculation) against
MetPy reference implementation to ensure accuracy and correctness.
"""

import pytest
import numpy as np
import warnings

# Import the function to test
from meteocalc.thermodynamics._jit_equations import _poisson_scalar  # Adjust import path

# Import reference implementation
try:
    from metpy.calc import potential_temperature
    from metpy.units import units
    HAS_METPY = True
except ImportError:
    HAS_METPY = False
    warnings.warn("MetPy not installed - skipping MetPy validation tests")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def standard_conditions():
    """Standard atmospheric conditions."""
    return {
        'temp_k': 288.15,  # 15°C
        'p': 1000.0,       # 1000 hPa
        'p0': 1000.0,      # Reference pressure
    }


@pytest.fixture
def upper_air_conditions():
    """Typical upper air sounding conditions."""
    return {
        'temp_k': 273.15,  # 0°C
        'p': 500.0,        # 500 hPa
        'p0': 1000.0,
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestBasicFunctionality:
    """Test basic function behavior and return types."""

    def test_returns_float(self, standard_conditions):
        """Function should return float."""
        theta = _poisson_scalar(
            standard_conditions['temp_k'],
            standard_conditions['p'],
            standard_conditions['p0']
        )
        
        assert isinstance(theta, (float, np.floating)), \
            f"Expected float, got {type(theta)}"
        assert np.isfinite(theta), "Result should be finite"
        assert theta > 0, "Potential temperature must be positive"

    def test_at_reference_pressure(self, standard_conditions):
        """At reference pressure, theta should equal temperature."""
        theta = _poisson_scalar(
            standard_conditions['temp_k'],
            standard_conditions['p'],
            standard_conditions['p0']
        )
        
        # At p = p0, theta = T
        assert abs(theta - standard_conditions['temp_k']) < 0.01, \
            f"At reference pressure, theta ({theta}K) should equal T ({standard_conditions['temp_k']}K)"

    def test_result_in_kelvin(self):
        """Result should be in Kelvin (positive value)."""
        theta = _poisson_scalar(temp_k=288.15, p=850.0, p0=1000.0)
        
        assert theta > 0, "Temperature must be positive (Kelvin)"
        assert theta < 1000, "Unrealistic temperature (sanity check)"


# ============================================================================
# MetPy Validation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestMetPyValidation:
    """Validate against MetPy implementation."""

    def test_against_metpy_scalar(self, standard_conditions):
        """Compare with MetPy for scalar inputs."""
        temp_k = standard_conditions['temp_k']
        p = standard_conditions['p']
        p0 = standard_conditions['p0']

        # Our implementation
        theta_ours = _poisson_scalar(temp_k, p, p0)

        # MetPy implementation
        temp_metpy = temp_k * units.K
        p_metpy = p * units.hPa
        theta_metpy = potential_temperature(p_metpy, temp_metpy)
        theta_metpy_k = theta_metpy.to('K').magnitude

        error = abs(theta_ours - theta_metpy_k)
        assert error < 0.2, \
            f"MetPy validation failed:\n" \
            f"  Ours:  {theta_ours:.4f}K\n" \
            f"  MetPy: {theta_metpy_k:.4f}K\n" \
            f"  Error: {error:.4f}K"

    @pytest.mark.parametrize("temp_c,p,p0", [
        (15.0, 1000.0, 1000.0),   # Sea level
        (15.0, 850.0, 1000.0),    # 850 hPa
        (0.0, 500.0, 1000.0),     # 500 hPa
        (-40.0, 250.0, 1000.0),   # 250 hPa
        (25.0, 1013.25, 1000.0),  # Standard sea level
        (10.0, 700.0, 1000.0),    # 700 hPa
        (-20.0, 300.0, 1000.0),   # Upper troposphere
        (-55.0, 200.0, 1000.0),   # Near tropopause
        (30.0, 925.0, 1000.0),    # Warm boundary layer
        (-60.0, 100.0, 1000.0),   # Lower stratosphere
    ])
    def test_against_metpy_parametrized(self, temp_c, p, p0):
        """Parametrized comparison with MetPy across various conditions."""
        temp_k = temp_c + 273.15

        # Our implementation
        theta_ours = _poisson_scalar(temp_k, p, p0)

        # MetPy implementation
        theta_metpy = potential_temperature(
            pressure=p * units.hPa,
            temperature=temp_k * units.K
        ).to('K').magnitude

        error = abs(theta_ours - theta_metpy)
        assert error < 0.2, \
            f"At T={temp_c}°C, P={p} hPa, P0={p0} hPa:\n" \
            f"  Ours:  {theta_ours:.4f}K\n" \
            f"  MetPy: {theta_metpy:.4f}K\n" \
            f"  Error: {error:.4f}K"

    def test_against_metpy_array_conditions(self):
        """Compare with MetPy for array of conditions."""
        temps_k = np.array([273.15, 283.15, 293.15, 303.15])  # 0, 10, 20, 30°C
        pressures = np.array([1000.0, 850.0, 700.0, 500.0])   # Various levels
        p0 = 1000.0

        # Our implementation (element-wise)
        thetas_ours = np.array([
            _poisson_scalar(t, p, p0)
            for t, p in zip(temps_k, pressures)
        ])

        # MetPy implementation (element-wise)
        thetas_metpy = np.array([
            potential_temperature(
                p * units.hPa,
                t * units.K
            ).to('K').magnitude
            for t, p in zip(temps_k, pressures)
        ])

        errors = np.abs(thetas_ours - thetas_metpy)
        max_error = np.max(errors)

        assert max_error < 0.2, \
            f"MetPy array validation failed:\n" \
            f"  Ours:      {thetas_ours}\n" \
            f"  MetPy:     {thetas_metpy}\n" \
            f"  Max error: {max_error:.4f}K"

    def test_against_metpy_different_reference_pressures(self):
        """Test with various reference pressures against MetPy."""
        temp_k = 288.15
        p = 850.0
        
        reference_pressures = [1000.0, 1000.0, 1000.0, 1000.0]

        for p0 in reference_pressures:
            # Our implementation
            theta_ours = _poisson_scalar(temp_k, p, p0)

            # MetPy implementation
            theta_metpy = potential_temperature(
                p * units.hPa,
                temp_k * units.K
            ).to('K').magnitude

            error = abs(theta_ours - theta_metpy)
            assert error < 0.01, \
                f"Reference pressure P0={p0} hPa:\n" \
                f"  Ours:  {theta_ours:.4f}K\n" \
                f"  MetPy: {theta_metpy:.4f}K\n" \
                f"  Error: {error:.4f}K"

    def test_against_metpy_extreme_cold(self):
        """Test extreme cold conditions against MetPy."""
        extreme_cold_cases = [
            (180.0, 100.0, 1000.0),  # Very cold, low pressure
            (200.0, 250.0, 1000.0),  # Cold, mid pressure
            (220.0, 500.0, 1000.0),  # Cold, mid-high pressure
        ]

        for temp_k, p, p0 in extreme_cold_cases:
            theta_ours = _poisson_scalar(temp_k, p, p0)
            
            theta_metpy = potential_temperature(
                p * units.hPa,
                temp_k * units.K
            ).to('K').magnitude

            error = abs(theta_ours - theta_metpy)
            assert error < 0.2, \
                f"Extreme cold T={temp_k}K, P={p} hPa:\n" \
                f"  Ours:  {theta_ours:.4f}K\n" \
                f"  MetPy: {theta_metpy:.4f}K"

    def test_against_metpy_extreme_heat(self):
        """Test extreme heat conditions against MetPy."""
        extreme_heat_cases = [
            (320.0, 1000.0, 1000.0),  # Very hot, sea level
            (350.0, 850.0, 1000.0),   # Extremely hot
        ]

        for temp_k, p, p0 in extreme_heat_cases:
            theta_ours = _poisson_scalar(temp_k, p, p0)
            
            theta_metpy = potential_temperature(
                p * units.hPa,
                temp_k * units.K
            ).to('K').magnitude

            error = abs(theta_ours - theta_metpy)
            assert error < 0.2, \
                f"Extreme heat T={temp_k}K, P={p} hPa:\n" \
                f"  Ours:  {theta_ours:.4f}K\n" \
                f"  MetPy: {theta_metpy:.4f}K"

    def test_against_metpy_vertical_profile(self):
        """Test realistic vertical atmospheric profile against MetPy."""
        # Typical sounding from surface to tropopause
        pressures = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200])  # hPa
        temps_c = np.array([15, 10, 5, -5, -20, -30, -45, -55, -60])  # °C
        temps_k = temps_c + 273.15
        p0 = 1000.0

        # Our implementation
        thetas_ours = np.array([
            _poisson_scalar(t, p, p0)
            for t, p in zip(temps_k, pressures)
        ])

        # MetPy implementation
        thetas_metpy = np.array([
            potential_temperature(
                p * units.hPa,
                t * units.K
            ).to('K').magnitude
            for t, p in zip(temps_k, pressures)
        ])

        errors = np.abs(thetas_ours - thetas_metpy)
        max_error = np.max(errors)

        assert max_error < 0.2, \
            f"Vertical profile validation failed:\n" \
            f"  Max error: {max_error:.4f}K\n" \
            f"  Errors: {errors}"

    def test_against_metpy_boundary_layer(self):
        """Test boundary layer conditions against MetPy."""
        # Typical boundary layer conditions
        boundary_layer_cases = [
            (298.15, 1000.0, 1000.0),  # Warm surface
            (295.15, 990.0, 1000.0),   # Just above surface
            (290.15, 950.0, 1000.0),   # Mid boundary layer
            (285.15, 925.0, 1000.0),   # Top of boundary layer
        ]

        for temp_k, p, p0 in boundary_layer_cases:
            theta_ours = _poisson_scalar(temp_k, p, p0)
            
            theta_metpy = potential_temperature(
                p * units.hPa,
                temp_k * units.K
            ).to('K').magnitude

            error = abs(theta_ours - theta_metpy)
            assert error < 0.2, \
                f"Boundary layer T={temp_k}K, P={p} hPa:\n" \
                f"  Ours:  {theta_ours:.4f}K\n" \
                f"  MetPy: {theta_metpy:.4f}K"


# ============================================================================
# Physical Correctness Tests
# ============================================================================

class TestPhysicalCorrectness:
    """Test physical relationships and constraints."""

    def test_theta_increases_with_decreasing_pressure(self):
        """Potential temp should increase as pressure decreases (at constant T)."""
        temp_k = 288.15
        p0 = 1000.0

        theta_1000 = _poisson_scalar(temp_k, 1000.0, p0)
        theta_850 = _poisson_scalar(temp_k, 850.0, p0)
        theta_500 = _poisson_scalar(temp_k, 500.0, p0)

        assert theta_850 > theta_1000, \
            "Theta should increase as pressure decreases"
        assert theta_500 > theta_850, \
            "Theta should increase as pressure decreases"

    def test_theta_increases_with_temperature(self):
        """Potential temp should increase with temperature (at constant P)."""
        p = 850.0
        p0 = 1000.0

        theta_cold = _poisson_scalar(263.15, p, p0)  # -10°C
        theta_warm = _poisson_scalar(293.15, p, p0)  # +20°C

        assert theta_warm > theta_cold, \
            f"Warmer air should have higher theta: {theta_warm} vs {theta_cold}"

    def test_theta_equals_temp_at_reference_pressure(self):
        """At reference pressure, theta = T."""
        temp_k = 288.15
        p0 = 1000.0

        theta = _poisson_scalar(temp_k, p0, p0)

        assert abs(theta - temp_k) < 0.01, \
            f"At P=P0, theta ({theta}K) should equal T ({temp_k}K)"

    def test_theta_always_positive(self):
        """Potential temperature must always be positive (Kelvin)."""
        test_cases = [
            (273.15, 1000.0, 1000.0),  # 0°C at sea level
            (233.15, 250.0, 1000.0),   # -40°C at 250 hPa
            (220.0, 100.0, 1000.0),    # Very cold, very high
        ]

        for temp_k, p, p0 in test_cases:
            theta = _poisson_scalar(temp_k, p, p0)
            assert theta > 0, \
                f"Theta must be positive, got {theta}K at T={temp_k}K, P={p} hPa"


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_pressure(self):
        """Test at very low pressure (high altitude)."""
        theta = _poisson_scalar(temp_k=220.0, p=10.0, p0=1000.0)

        assert np.isfinite(theta), "Should handle very low pressure"
        assert theta > 220.0, "Theta should be higher than T at low pressure"
        assert theta < 1000.0, "Result should be realistic"

    def test_very_high_pressure(self):
        """Test at very high pressure (below sea level)."""
        theta = _poisson_scalar(temp_k=300.0, p=1100.0, p0=1000.0)

        assert np.isfinite(theta), "Should handle high pressure"
        assert theta < 300.0, "Theta should be lower than T at high pressure"

    def test_different_reference_pressures(self):
        """Test with different reference pressures."""
        temp_k = 288.15
        p = 850.0

        theta_1000 = _poisson_scalar(temp_k, p, 1000.0)
        theta_500 = _poisson_scalar(temp_k, p, 500.0)

        # Different reference pressures should give different results
        assert theta_1000 != theta_500, \
            "Different reference pressures should yield different theta"


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_precision_consistency(self):
        """Test that small changes in input produce small changes in output."""
        temp_k = 288.15
        p = 850.0
        p0 = 1000.0

        theta1 = _poisson_scalar(temp_k, p, p0)
        theta2 = _poisson_scalar(temp_k + 0.01, p, p0)  # +0.01K

        # Small input change should produce small output change
        diff = abs(theta2 - theta1)
        assert diff < 0.1, \
            f"Small input change produced large output change: {diff}K"

    def test_floating_point_accuracy(self):
        """Test floating point accuracy at various magnitudes."""
        test_cases = [
            (288.15, 1000.0, 1000.0),
            (288.15, 100.0, 1000.0),
            (288.15, 10.0, 1000.0),
        ]

        for temp_k, p, p0 in test_cases:
            theta = _poisson_scalar(temp_k, p, p0)
            assert np.isfinite(theta), \
                f"Lost precision at P={p} hPa"


# ============================================================================
# Summary Report
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
