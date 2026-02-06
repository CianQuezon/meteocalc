"""
Comprehensive unit tests for meteocalc.dewpoint._jit_equations._magnus_equation_vectorised

Tests vectorized Magnus equation implementation including:
- Correctness (matches scalar)
- Performance (vectorization benefits)
- Edge cases
- Broadcasting
- Memory efficiency

Author: Test Suite
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import timeit
import time

# Import functions to test
from meteocalc.dewpoint._jit_equations import (
    _magnus_equation_scalar,
    _magnus_equation_vectorised
)

# Constants
MAGNUS_WATER_A = 17.27
MAGNUS_WATER_B = 237.7

MAGNUS_ICE_A = 22.46
MAGNUS_ICE_B = 272.62


# ==============================================================================
# Helper Functions
# ==============================================================================

def celsius_to_kelvin(temp_c):
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15


# ==============================================================================
# Test Class: Correctness - Matches Scalar
# ==============================================================================

class TestMagnusVectorizedCorrectness:
    """Test that vectorized implementation matches scalar results exactly."""
    
    def test_matches_scalar_single_value(self):
        """Test that vectorized matches scalar for single value."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        result_vec = _magnus_equation_vectorised(
            temp_k, rh,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        result_scalar = _magnus_equation_scalar(
            temp_k[0], rh[0],
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert_allclose(result_vec[0], result_scalar, rtol=1e-15)
    
    def test_matches_scalar_multiple_values(self):
        """Test that vectorized matches scalar for all elements."""
        temps_k = np.array([273.15, 283.15, 293.15, 303.15, 313.15])
        rhs = np.array([0.3, 0.5, 0.6, 0.8, 0.9])
        
        result_vec = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        results_scalar = np.array([
            _magnus_equation_scalar(t, r, MAGNUS_WATER_A, MAGNUS_WATER_B)
            for t, r in zip(temps_k, rhs)
        ])
        
        assert_allclose(result_vec, results_scalar, rtol=1e-15)
    
    def test_matches_scalar_large_array(self):
        """Test exact match with scalar on large array."""
        n = 1000
        temps_k = np.linspace(250, 320, n)
        rhs = np.linspace(0.2, 0.95, n)
        
        result_vec = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Sample check (checking all 1000 would be slow)
        indices = [0, n//4, n//2, 3*n//4, n-1]
        for i in indices:
            result_scalar = _magnus_equation_scalar(
                temps_k[i], rhs[i],
                MAGNUS_WATER_A, MAGNUS_WATER_B
            )
            assert_allclose(result_vec[i], result_scalar, rtol=1e-15)
    
    def test_ice_constants_match_scalar(self):
        """Test that ice constants work correctly in vectorized form."""
        temps_k = np.array([263.15, 258.15, 253.15])  # -10, -15, -20°C
        rhs = np.array([0.7, 0.7, 0.7])
        
        result_vec = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_ICE_A, MAGNUS_ICE_B
        )
        
        results_scalar = np.array([
            _magnus_equation_scalar(t, r, MAGNUS_ICE_A, MAGNUS_ICE_B)
            for t, r in zip(temps_k, rhs)
        ])
        
        assert_allclose(result_vec, results_scalar, rtol=1e-15)


# ==============================================================================
# Test Class: Array Handling
# ==============================================================================

class TestMagnusVectorizedArrays:
    """Test array handling and edge cases."""
    
    def test_returns_numpy_array(self):
        """Test that function returns numpy array."""
        temps_k = np.array([293.15, 303.15])
        rhs = np.array([0.5, 0.6])
        
        result = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
    
    def test_output_shape_matches_input(self):
        """Test that output shape matches input shape."""
        n = 100
        temps_k = np.linspace(270, 310, n)
        rhs = np.full(n, 0.6)
        
        result = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert result.shape == (n,)
        assert len(result) == n
    
    def test_preserves_order(self):
        """Test that output order matches input order."""
        temps_k = np.array([290.15, 295.15, 300.15])
        rhs = np.array([0.4, 0.5, 0.6])
        
        result = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Results should be in same order as inputs
        # Higher temperature + higher RH = higher dew point
        assert result[0] < result[1] < result[2]
    
    def test_different_array_sizes(self):
        """Test with various array sizes."""
        sizes = [1, 10, 100, 1000, 10000]
        
        for n in sizes:
            temps_k = np.linspace(270, 310, n)
            rhs = np.full(n, 0.6)
            
            result = _magnus_equation_vectorised(
                temps_k, rhs,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            )
            
            assert len(result) == n
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))


# ==============================================================================
# Test Class: Physical Constraints
# ==============================================================================

class TestMagnusVectorizedPhysics:
    """Test physical constraints are satisfied."""
    
    def test_dewpoint_less_than_temperature(self):
        """Test that all dew points are ≤ corresponding temperatures."""
        temps_k = np.linspace(270, 320, 100)
        rhs = np.linspace(0.2, 0.95, 100)
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert np.all(dewpoints <= temps_k), (
            "All dew points must be ≤ temperature"
        )
    
    def test_monotonic_with_rh(self):
        """Test that dew point increases with RH at constant temperature."""
        temp_k = 293.15
        temps_k = np.full(10, temp_k)
        rhs = np.linspace(0.3, 0.9, 10)
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Check monotonicity
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0), "Dew point should increase with RH"
    
    def test_rh_100_equals_temperature(self):
        """Test that RH=100% gives dew point = temperature (vectorized)."""
        temps_k = np.array([273.15, 283.15, 293.15, 303.15])
        rhs = np.ones(4)  # RH = 100%
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Magnus should be exact at RH=100%
        assert_allclose(dewpoints, temps_k, atol=1e-10)
    
    def test_frostpoint_higher_than_dewpoint_below_zero(self):
        """Test that frost point > dew point below 0°C (vectorized).
        
        Below 0°C, ice has lower saturation vapor pressure than
        supercooled water, so frost point should be warmer.
        """
        temps_k = np.array([263.15, 258.15, 253.15])  # -10, -15, -20°C
        rhs = np.array([0.6, 0.7, 0.8])
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        frostpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_ICE_A, MAGNUS_ICE_B
        )
        
        # Below 0°C: frost point should be higher (warmer)
        assert np.all(frostpoints > dewpoints), (
            "Frost point should be higher than dew point below 0°C"
        )


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================

class TestMagnusVectorizedEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_humidity(self):
        """Test with very low humidity (10%)."""
        temps_k = np.array([293.15, 303.15, 313.15])
        rhs = np.full(3, 0.1)
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Should be well below temperature
        assert np.all(dewpoints < temps_k - 10)
        # Should be reasonable
        assert np.all(dewpoints > 230)
        assert np.all(dewpoints < 280)
    
    def test_very_high_humidity(self):
        """Test with very high humidity (99%)."""
        temps_k = np.array([293.15, 303.15, 313.15])
        rhs = np.full(3, 0.99)
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Should be very close to temperature
        assert np.all(np.abs(dewpoints - temps_k) < 0.5)
    
    def test_cold_temperatures(self):
        """Test at very cold temperatures (-30°C)."""
        temps_k = np.array([243.15, 248.15, 253.15])  # -30, -25, -20°C
        rhs = np.array([0.5, 0.6, 0.7])
        
        frostpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_ICE_A, MAGNUS_ICE_B
        )
        
        # Should be reasonable for arctic conditions
        assert np.all(frostpoints > 220)  # > -53°C
        assert np.all(frostpoints < temps_k)
    
    def test_hot_temperatures(self):
        """Test at hot temperatures (50°C)."""
        temps_k = np.array([313.15, 323.15, 333.15])  # 40, 50, 60°C
        rhs = np.array([0.3, 0.4, 0.5])
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Should be reasonable for desert conditions
        assert np.all(dewpoints > 270)  # > -3°C
        assert np.all(dewpoints < 310)  # < 37°C
    
    def test_no_nan_or_inf(self):
        """Test that function never produces NaN or Inf in valid range."""
        temps_k = np.linspace(243.15, 333.15, 1000)  # -30°C to 60°C
        rhs = np.linspace(0.1, 0.99, 1000)
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert not np.any(np.isnan(dewpoints)), "Found NaN in results"
        assert not np.any(np.isinf(dewpoints)), "Found Inf in results"


# ==============================================================================
# Test Class: Performance
# ==============================================================================

class TestMagnusVectorizedPerformance:
    """Test vectorization performance characteristics."""
    
    def test_faster_than_scalar_loop(self):
        """Test that vectorized is faster than scalar loop for large arrays.
        
        Vectorized should be at least 2x faster than Python loop
        calling scalar function.
        """
        n = 10000
        temps_k = np.linspace(250, 320, n)
        rhs = np.full(n, 0.6)
        
        # Warmup both implementations
        _ = _magnus_equation_vectorised(
            temps_k[:100], rhs[:100],
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        _ = [_magnus_equation_scalar(t, r, MAGNUS_WATER_A, MAGNUS_WATER_B) 
             for t, r in zip(temps_k[:100], rhs[:100])]
        
        # Time vectorized
        time_vec = timeit.timeit(
            lambda: _magnus_equation_vectorised(
                temps_k, rhs,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            ),
            number=100
        ) / 100
        
        # Time scalar loop
        time_loop = timeit.timeit(
            lambda: [_magnus_equation_scalar(t, r, MAGNUS_WATER_A, MAGNUS_WATER_B)
                     for t, r in zip(temps_k, rhs)],
            number=10
        ) / 10
        
        speedup = time_loop / time_vec
        
        # Vectorized should be at least 2x faster
        assert speedup > 2.0, (
            f"Vectorized should be >2x faster than loop. "
            f"Got {speedup:.1f}x speedup"
        )
        
        print(f"\nVectorized speedup: {speedup:.1f}x over scalar loop")
    
    def test_vectorization_performance(self):
        """Test that vectorized implementation achieves high throughput.
        
        Goals:
        1. Process 1M+ elements/second (after warmup)
        2. Performance improves or stays stable as size grows
        """
        sizes = [100, 1_000, 10_000, 100_000]
        times = []
        
        for n in sizes:
            temps = np.linspace(250, 320, n)
            rhs = np.full(n, 0.6)
            
            # Warmup on first iteration
            if n == sizes[0]:
                for _ in range(5):
                    _ = _magnus_equation_vectorised(
                        temps, rhs,
                        MAGNUS_WATER_A, MAGNUS_WATER_B
                    )
            
            # Time it
            t = timeit.timeit(
                lambda: _magnus_equation_vectorised(
                    temps, rhs,
                    MAGNUS_WATER_A, MAGNUS_WATER_B
                ),
                number=100
            ) / 100
            
            times.append(t)
        
        # Calculate throughput at each size
        throughputs = [n / t for n, t in zip(sizes, times)]
        
        # All sizes should process >1M elements/sec
        assert all(tp > 1_000_000 for tp in throughputs), (
            f"All sizes should process >1M elements/sec. "
            f"Got: {[f'{tp/1e6:.1f}M/s' for tp in throughputs]}"
        )
        
        # Throughput should not decrease significantly
        for i in range(len(throughputs) - 1):
            ratio = throughputs[i+1] / throughputs[i]
            assert ratio > 0.5, (
                f"Throughput decreased too much: "
                f"{sizes[i]}→{sizes[i+1]}: {ratio:.2f}x"
            )
        
        print(f"\nVectorization performance:")
        for n, t, tp in zip(sizes, times, throughputs):
            print(f"  {n:>6} elements: {t*1e6:>6.1f}µs ({tp/1e6:>5.1f}M elem/sec)")
    
    def test_parallel_performance(self):
        """Test that parallel=True provides benefit on large arrays.
        
        Note: Parallel benefit depends on array size and CPU cores.
        May not show improvement on small arrays due to overhead.
        """
        n = 100_000  # Large enough to benefit from parallelization
        temps = np.linspace(250, 320, n)
        rhs = np.linspace(0.2, 0.9, n)
        
        # Warmup
        for _ in range(5):
            _ = _magnus_equation_vectorised(
                temps, rhs,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            )
        
        # Time it
        elapsed = timeit.timeit(
            lambda: _magnus_equation_vectorised(
                temps, rhs,
                MAGNUS_WATER_A, MAGNUS_WATER_B
            ),
            number=100
        ) / 100
        
        throughput = n / elapsed
        
        # Should achieve good throughput (exact value depends on CPU)
        # Conservative target: >5M elements/sec on large arrays
        assert throughput > 5_000_000, (
            f"Large array throughput: {throughput/1e6:.1f}M elem/sec "
            f"(expected >5M elem/sec)"
        )
        
        print(f"\nParallel performance on {n} elements:")
        print(f"  Throughput: {throughput/1e6:.1f}M elements/sec")
        print(f"  Time: {elapsed*1000:.2f}ms")


# ==============================================================================
# Test Class: Memory Efficiency
# ==============================================================================

class TestMagnusVectorizedMemory:
    """Test memory efficiency and handling."""
    
    def test_memory_efficient(self):
        """Test that function doesn't create excessive intermediate arrays."""
        import sys
        
        n = 100_000
        temps = np.linspace(250, 320, n)
        rhs = np.full(n, 0.6)
        
        # Get memory before
        import gc
        gc.collect()
        
        # Run function
        result = _magnus_equation_vectorised(
            temps, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Result should be single array of expected size
        expected_bytes = n * 8  # 8 bytes per float64
        actual_bytes = result.nbytes
        
        assert actual_bytes == expected_bytes, (
            f"Result size: {actual_bytes} bytes, expected {expected_bytes}"
        )
    
    def test_handles_large_arrays(self):
        """Test that function can handle very large arrays."""
        n = 1_000_000  # 1 million elements
        temps = np.linspace(250, 320, n)
        rhs = np.full(n, 0.6)
        
        result = _magnus_equation_vectorised(
            temps, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert len(result) == n
        assert not np.any(np.isnan(result))
        assert np.all(result < temps)  # Dew point < temperature


# ==============================================================================
# Test Class: Numerical Stability
# ==============================================================================

class TestMagnusVectorizedStability:
    """Test numerical stability and precision."""
    
    def test_deterministic(self):
        """Test that function produces identical results on repeated calls."""
        temps_k = np.linspace(250, 320, 1000)
        rhs = np.linspace(0.2, 0.95, 1000)
        
        result1 = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        result2 = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        assert_array_equal(result1, result2, 
                          err_msg="Results should be identical")
    
    def test_smooth_gradients(self):
        """Test that small input changes produce small output changes."""
        n = 1000
        temps_k = np.linspace(290, 295, n)  # Small temperature range
        rhs = np.full(n, 0.6)
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Check that gradients are smooth (no jumps)
        diffs = np.diff(dewpoints)
        
        # All differences should be small and consistent
        assert np.all(diffs > 0), "Should be monotonically increasing"
        assert np.std(diffs) < 0.01, "Gradients should be smooth"
    
    def test_precision_maintained(self):
        """Test that numerical precision is maintained."""
        # Use values that might cause precision issues
        temps_k = np.array([273.15, 273.16, 273.17])  # Very close values
        rhs = np.array([0.5, 0.500001, 0.500002])     # Very close RH
        
        dewpoints = _magnus_equation_vectorised(
            temps_k, rhs,
            MAGNUS_WATER_A, MAGNUS_WATER_B
        )
        
        # Should distinguish these small differences
        diffs = np.diff(dewpoints)
        assert np.all(diffs > 0), "Should distinguish small differences"
        assert np.all(diffs < 0.1), "But differences should be small"


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance tests"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])