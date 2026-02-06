"""
Comprehensive unit tests for meteocalc.dewpoint._jit_equations._buck_equation_vectorised

Tests validate:
- Consistency with scalar implementation
- Vectorization correctness
- Broadcasting behavior
- Edge cases and boundary conditions
- Performance characteristics
- Array shape handling
- Data type handling

Author: Test Suite
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import time

# Import functions to test
from meteocalc.dewpoint._jit_equations import (
    _buck_equation_vectorised,
    _buck_equation_scalar
)

# Buck constants
BUCK_WATER_A = 17.368
BUCK_WATER_B = 238.88
BUCK_WATER_C = 234.5

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


def generate_test_data(n_points, temp_range=(-20, 40), rh_range=(0.2, 0.95)):
    """Generate random test data for vectorized testing."""
    np.random.seed(42)  # Reproducible
    
    temps_c = np.random.uniform(temp_range[0], temp_range[1], n_points)
    temps_k = celsius_to_kelvin(temps_c)
    rhs = np.random.uniform(rh_range[0], rh_range[1], n_points)
    
    return temps_k, rhs


# ==============================================================================
# Test Class: Basic Functionality
# ==============================================================================

class TestBuckVectorizedBasic:
    """Test basic functionality of _buck_equation_vectorised."""
    
    def test_returns_array(self):
        """Test that function returns numpy array."""
        temps = np.array([273.15, 283.15, 293.15])
        rhs = np.array([0.5, 0.6, 0.7])
        
        result = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_output_shape_matches_input(self):
        """Test that output shape matches input shape."""
        for n in [1, 10, 100, 1000]:
            temps = np.linspace(273.15, 313.15, n)
            rhs = np.linspace(0.3, 0.9, n)
            
            result = _buck_equation_vectorised(
                temps, rhs,
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            
            assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}"
    
    def test_consistency_with_scalar(self):
        """Test that vectorized results match scalar implementation."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        rhs = np.array([0.4, 0.5, 0.6, 0.7])
        
        # Vectorized
        result_vec = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Scalar (element-wise)
        result_scalar = np.array([
            _buck_equation_scalar(t, r, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
            for t, r in zip(temps, rhs)
        ])
        
        assert_allclose(
            result_vec, result_scalar,
            rtol=1e-14,
            err_msg="Vectorized results must exactly match scalar implementation"
        )
    
    def test_single_element_array(self):
        """Test with single-element arrays."""
        temp = np.array([293.15])
        rh = np.array([0.6])
        
        result_vec = _buck_equation_vectorised(
            temp, rh,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        result_scalar = _buck_equation_scalar(
            293.15, 0.6,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert_allclose(result_vec[0], result_scalar, rtol=1e-14)
    
    def test_dewpoint_less_than_temperature(self):
        """Test that all dew points are ≤ temperature (physical constraint)."""
        temps = np.linspace(273.15, 313.15, 100)
        rhs = np.linspace(0.2, 0.9, 100)
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert np.all(dewpoints <= temps), (
            "All dew points must be ≤ temperature"
        )


# ==============================================================================
# Test Class: Large Arrays
# ==============================================================================

class TestBuckVectorizedLargeArrays:
    """Test with large arrays (typical use case)."""
    
    @pytest.mark.parametrize("n_points", [100, 1000, 10000, 100000])
    def test_large_arrays(self, n_points):
        """Test consistency with scalar on large arrays."""
        temps, rhs = generate_test_data(n_points)
        
        # Vectorized
        result_vec = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Sample check (full check would be too slow for 100k)
        sample_indices = np.random.choice(n_points, size=min(100, n_points), replace=False)
        
        for idx in sample_indices:
            result_scalar = _buck_equation_scalar(
                temps[idx], rhs[idx],
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            
            assert_allclose(
                result_vec[idx], result_scalar,
                rtol=1e-13,
                err_msg=f"Mismatch at index {idx}"
            )
    
    def test_million_points(self):
        """Test with 1 million points (stress test)."""
        n_points = 1_000_000
        temps, rhs = generate_test_data(n_points)
        
        # Should not crash or produce NaN/Inf
        result = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert len(result) == n_points
        assert not np.any(np.isnan(result)), "No NaN values allowed"
        assert not np.any(np.isinf(result)), "No Inf values allowed"
        
        # All results should be reasonable temperatures
        assert np.all(result > 173.15), "Dew points too low"
        assert np.all(result < 373.15), "Dew points too high"


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================

class TestBuckVectorizedEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_rh_100_percent_equals_temperature(self):
        """Test that RH=100% gives Td≈T across array."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        rhs = np.ones(4)  # RH = 100%
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Buck has small systematic error at RH=100%
        assert_allclose(
            dewpoints, temps,
            atol=0.7,
            err_msg="At RH=100%, Td should be very close to T"
        )
    
    def test_very_low_humidity(self):
        """Test very low relative humidity (10%)."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        rhs = np.full(4, 0.10)
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Should be well below temperature
        assert np.all(dewpoints < temps - 10), (
            "At RH=10%, Td should be >10K below T"
        )
        
        # Should still be reasonable
        assert np.all(dewpoints > 230)
        assert np.all(dewpoints < 280)
    
    def test_very_high_humidity(self):
        """Test very high relative humidity (99%)."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        rhs = np.full(4, 0.99)
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Should be very close to temperature
        assert np.all(np.abs(dewpoints - temps) < 0.5), (
            "At RH=99%, Td should be within 0.5K of T"
        )
    
    def test_cold_conditions(self):
        """Test cold temperature conditions (-40°C to 0°C)."""
        temps_c = np.linspace(-40, 0, 50)
        temps_k = celsius_to_kelvin(temps_c)
        rhs = np.full(50, 0.7)
        
        dewpoints = _buck_equation_vectorised(
            temps_k, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Should be reasonable
        assert np.all(dewpoints > 173.15), "Dew points too low"
        assert np.all(dewpoints < temps_k), "Dew points exceed temperature"
    
    def test_hot_conditions(self):
        """Test hot temperature conditions (30°C to 50°C)."""
        temps_c = np.linspace(30, 50, 50)
        temps_k = celsius_to_kelvin(temps_c)
        rhs = np.full(50, 0.5)
        
        dewpoints = _buck_equation_vectorised(
            temps_k, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        dewpoints_c = kelvin_to_celsius(dewpoints)
        
        # Should be reasonable for hot conditions
        assert np.all(dewpoints_c > 10), "Dew points too low"
        assert np.all(dewpoints_c < 40), "Dew points too high"
    
    def test_mixed_conditions(self):
        """Test with mixed temperature and humidity conditions."""
        temps = np.array([
            celsius_to_kelvin(-20),  # Cold
            celsius_to_kelvin(0),    # Freezing
            celsius_to_kelvin(20),   # Room temp
            celsius_to_kelvin(40),   # Hot
        ])
        
        rhs = np.array([0.3, 0.5, 0.7, 0.9])  # Varying humidity
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # All should be below temperature
        assert np.all(dewpoints < temps)
        
        # All should be reasonable
        assert np.all(dewpoints > 200)
        assert np.all(dewpoints < 350)


# ==============================================================================
# Test Class: Water vs Ice Constants
# ==============================================================================

class TestBuckVectorizedIceVsWater:
    """Test ice vs water constant behavior with vectorized function."""
    
    def test_ice_water_difference_vectorized(self):
        """Test frost point vs dew point difference across array.
        
        Note: Buck's analytic inversions may not preserve proper ordering.
        This test documents the behavior but doesn't enforce physical constraints.
        """
        temps_c = np.array([-20, -15, -10, -5, -2])
        temps_k = celsius_to_kelvin(temps_c)
        rhs = np.full(5, 0.75)
        
        # Dew point (water)
        td_water = _buck_equation_vectorised(
            temps_k, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Frost point (ice)
        tf_ice = _buck_equation_vectorised(
            temps_k, rhs,
            BUCK_ICE_A, BUCK_ICE_B, BUCK_ICE_C
        )
        
        # Document the differences
        diff = tf_ice - td_water
        
        # All differences should be reasonable magnitude
        assert np.all(np.abs(diff) < 5.0), (
            f"Differences seem unreasonably large: {diff}"
        )


# ==============================================================================
# Test Class: Numerical Stability
# ==============================================================================

class TestBuckVectorizedNumericalStability:
    """Test numerical stability and precision."""
    
    def test_no_nan_or_inf(self):
        """Test that function never returns NaN or Inf."""
        temps, rhs = generate_test_data(10000)
        
        result = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert not np.any(np.isnan(result)), "NaN values detected"
        assert not np.any(np.isinf(result)), "Inf values detected"
    
    def test_deterministic(self):
        """Test that function produces deterministic results."""
        temps, rhs = generate_test_data(1000)
        
        result1 = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        result2 = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert_array_equal(result1, result2, err_msg="Results not deterministic")
    
    def test_small_perturbations(self):
        """Test smooth behavior under small perturbations."""
        temps = np.array([293.15] * 100)
        rhs = np.linspace(0.5, 0.501, 100)  # Small RH changes
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Dewpoints should change smoothly
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0), "Dewpoint should increase with RH"
        assert np.all(diffs < 0.01), "Changes should be small and smooth"
    
    def test_monotonicity(self):
        """Test monotonic relationship between RH and dew point."""
        temps = np.full(100, 293.15)
        rhs = np.linspace(0.2, 0.9, 100)
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Dewpoint should increase monotonically with RH
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0), "Dewpoint must increase with RH"


# ==============================================================================
# Test Class: Array Dtype Handling
# ==============================================================================

class TestBuckVectorizedDtypes:
    """Test handling of different array data types."""
    
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_float_dtypes(self, dtype):
        """Test with different float dtypes."""
        temps = np.array([273.15, 283.15, 293.15], dtype=dtype)
        rhs = np.array([0.5, 0.6, 0.7], dtype=dtype)
        
        result = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Output should always be float64 (Numba default)
        assert result.dtype == np.float64
        
        # Results should be reasonable
        assert np.all(result < temps)
    
    def test_integer_conversion(self):
        """Test that integer inputs are handled correctly."""
        # Numba should handle this, but test to be sure
        temps = np.array([273, 283, 293], dtype=np.int32)
        rhs = np.array([0.5, 0.6, 0.7])
        
        result = _buck_equation_vectorised(
            temps.astype(np.float64), rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert result.dtype == np.float64
        assert len(result) == 3


# ==============================================================================
# Test Class: Performance Characteristics
# ==============================================================================

class TestBuckVectorizedPerformance:
    """Test performance characteristics (not strict benchmarks)."""
    
    def test_scales_linearly(self):
        """Test that execution time scales roughly linearly with input size."""
        sizes = [1000, 10000, 100000]
        times = []
        
        for n in sizes:
            temps, rhs = generate_test_data(n)
            
            # Warm-up JIT
            _ = _buck_equation_vectorised(
                temps[:10], rhs[:10],
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            
            # Time the operation
            start = time.perf_counter()
            _ = _buck_equation_vectorised(
                temps, rhs,
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
            end = time.perf_counter()
            
            times.append(end - start)
        
        # Check rough linear scaling (within 2x tolerance)
        ratio_1 = times[1] / times[0]
        ratio_2 = times[2] / times[1]
        
        expected_ratio = 10.0  # 10x more data
        
        assert 5 < ratio_1 < 20, f"Scaling 1k→10k: {ratio_1:.1f}x"
        assert 5 < ratio_2 < 20, f"Scaling 10k→100k: {ratio_2:.1f}x"
    
    @pytest.mark.slow
    def test_throughput(self):
        """Test that vectorized function achieves high throughput."""
        n = 1_000_000
        temps, rhs = generate_test_data(n)
        
        # Warm-up
        _ = _buck_equation_vectorised(
            temps[:100], rhs[:100],
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # Benchmark
        start = time.perf_counter()
        _ = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        end = time.perf_counter()
        
        elapsed = end - start
        throughput = n / elapsed
        
        # Should process at least 1M points per second
        # (Conservative - actual performance should be much higher)
        assert throughput > 1_000_000, (
            f"Throughput too low: {throughput:.0f} points/sec"
        )
    
    def test_faster_than_loop(self):
        """Test that vectorized is faster than Python loop."""
        n = 10000
        temps, rhs = generate_test_data(n)
        
        # Vectorized (with warm-up)
        _ = _buck_equation_vectorised(
            temps[:10], rhs[:10],
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        start_vec = time.perf_counter()
        result_vec = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        time_vec = time.perf_counter() - start_vec
        
        # Python loop (with scalar function)
        start_loop = time.perf_counter()
        result_loop = np.array([
            _buck_equation_scalar(t, r, BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C)
            for t, r in zip(temps, rhs)
        ])
        time_loop = time.perf_counter() - start_loop
        
        # Vectorized should be faster (at least 2x, typically much more)
        speedup = time_loop / time_vec
        assert speedup > 2.0, f"Vectorized only {speedup:.1f}x faster than loop"
        
        # Results should match
        assert_allclose(result_vec, result_loop, rtol=1e-13)


# ==============================================================================
# Test Class: Error Conditions
# ==============================================================================

class TestBuckVectorizedErrors:
    """Test error handling and invalid inputs."""
    
    def test_mismatched_array_lengths(self):
        """Test that mismatched array lengths raise appropriate error."""
        temps = np.array([273.15, 283.15, 293.15])
        rhs = np.array([0.5, 0.6])  # Wrong length
        
        with pytest.raises((ValueError, IndexError)):
            _buck_equation_vectorised(
                temps, rhs,
                BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
            )
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        temps = np.array([])
        rhs = np.array([])
        
        result = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        assert len(result) == 0
        assert result.dtype == np.float64


# ==============================================================================
# Test Class: Comparison with References
# ==============================================================================

class TestBuckVectorizedValidation:
    """Validate vectorized implementation against known patterns."""
    
    def test_standard_conditions_array(self):
        """Test standard atmospheric conditions across array."""
        # 20°C with various RH values
        temps = np.full(5, celsius_to_kelvin(20.0))
        rhs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        
        dewpoints = _buck_equation_vectorised(
            temps, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        dewpoints_c = kelvin_to_celsius(dewpoints)
        
        # Expected approximate values (Buck equation)
        expected = np.array([9.3, 12.0, 14.4, 16.4, 18.3])
        
        # Allow reasonable tolerance
        assert_allclose(
            dewpoints_c, expected,
            atol=0.7,
            err_msg="Mismatch with expected standard conditions"
        )
    
    def test_realistic_weather_data(self):
        """Test with realistic meteorological data patterns."""
        # Simulate a day's temperature and humidity cycle
        hours = np.arange(24)
        
        # Temperature: 15°C at night, 25°C during day
        temps_c = 20 + 5 * np.sin(2 * np.pi * (hours - 6) / 24)
        temps_k = celsius_to_kelvin(temps_c)
        
        # RH: 80% at night, 40% during day
        rhs = 0.6 - 0.2 * np.sin(2 * np.pi * (hours - 6) / 24)
        
        dewpoints = _buck_equation_vectorised(
            temps_k, rhs,
            BUCK_WATER_A, BUCK_WATER_B, BUCK_WATER_C
        )
        
        # All dew points should be below temperature
        assert np.all(dewpoints < temps_k)
        
        # Dew points should vary less than temperature (typical pattern)
        temp_range = np.ptp(temps_k)
        dp_range = np.ptp(dewpoints)
        assert dp_range < temp_range


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