"""
Comprehensive tests for RootSolvers._create_substitute_results helper function.

Tests cover:
- Scalar vs array inputs
- Priority order (x0 > a > b)
- Different input types (float, int, list, ndarray)
- Edge cases (empty arrays, 0-D arrays, large arrays)
- Error handling (all None inputs)
- Data type correctness (float64, int64, bool)
- Shape consistency

Author: Test Suite
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from meteorological_equations.math.solvers.core import RootSolvers


class TestCreateSubstituteResultsScalar:
    """Test scalar (single value) inputs."""
    
    def test_scalar_with_x0_float(self):
        """Test scalar float input via x0."""
        roots, iters, conv = RootSolvers._create_substitute_results(x0=1.0)
        
        assert np.isnan(roots), "Root should be NaN"
        assert iters == 100, "Iterations should be 100"
        assert conv is False, "Converged should be False"
        
        # Check types
        assert isinstance(roots, float), "Root should be float"
        assert isinstance(iters, int), "Iterations should be int"
        assert isinstance(conv, bool), "Converged should be bool"
    
    def test_scalar_with_a_float(self):
        """Test scalar float input via a when x0 is None."""
        roots, iters, conv = RootSolvers._create_substitute_results(a=2.5)
        
        assert np.isnan(roots)
        assert iters == 100
        assert conv is False
    
    def test_scalar_with_b_float(self):
        """Test scalar float input via b when x0 and a are None."""
        roots, iters, conv = RootSolvers._create_substitute_results(b=3.0)
        
        assert np.isnan(roots)
        assert iters == 100
        assert conv is False
    
    def test_scalar_with_custom_max_iter(self):
        """Test scalar with custom max_iter value."""
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=1.0, max_iter=50
        )
        
        assert np.isnan(roots)
        assert iters == 50, "Should use custom max_iter"
        assert conv is False
    
    def test_scalar_with_zero_max_iter(self):
        """Test scalar with max_iter=0."""
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=1.0, max_iter=0
        )
        
        assert np.isnan(roots)
        assert iters == 0
        assert conv is False
    
    def test_scalar_with_negative_value(self):
        """Test scalar with negative input value."""
        roots, iters, conv = RootSolvers._create_substitute_results(x0=-5.0)
        
        assert np.isnan(roots)
        assert iters == 100
        assert conv is False
    
    def test_scalar_with_zero_value(self):
        """Test scalar with zero input value."""
        roots, iters, conv = RootSolvers._create_substitute_results(x0=0.0)
        
        assert np.isnan(roots)
        assert iters == 100
        assert conv is False
    
    def test_scalar_with_integer_input(self):
        """Test scalar integer input (should work like float)."""
        roots, iters, conv = RootSolvers._create_substitute_results(x0=5)
        
        assert np.isnan(roots)
        assert iters == 100
        assert conv is False


class TestCreateSubstituteResultsArray:
    """Test array (vectorized) inputs."""
    
    def test_array_with_x0_small(self):
        """Test small array input via x0."""
        x0 = np.array([1.0, 2.0, 3.0])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        # Check shapes
        assert roots.shape == (3,), "Roots shape should match input"
        assert iters.shape == (3,), "Iterations shape should match input"
        assert conv.shape == (3,), "Converged shape should match input"
        
        # Check values
        assert np.all(np.isnan(roots)), "All roots should be NaN"
        assert np.all(iters == 100), "All iterations should be 100"
        assert np.all(conv == False), "All converged should be False"
        
        # Check dtypes
        assert roots.dtype == np.float64, "Roots should be float64"
        assert iters.dtype == np.int64, "Iterations should be int64"
        assert conv.dtype == bool, "Converged should be bool"
    
    def test_array_with_a_medium(self):
        """Test medium array input via a when x0 is None."""
        a = np.linspace(0, 10, 20)
        roots, iters, conv = RootSolvers._create_substitute_results(a=a)
        
        assert roots.shape == (20,)
        assert iters.shape == (20,)
        assert conv.shape == (20,)
        
        assert np.all(np.isnan(roots))
        assert np.all(iters == 100)
        assert np.all(conv == False)
    
    def test_array_with_b_large(self):
        """Test large array input via b when x0 and a are None."""
        b = np.arange(1000)
        roots, iters, conv = RootSolvers._create_substitute_results(b=b)
        
        assert roots.shape == (1000,)
        assert iters.shape == (1000,)
        assert conv.shape == (1000,)
        
        assert np.all(np.isnan(roots))
        assert np.all(iters == 100)
        assert np.all(conv == False)
    
    def test_array_with_custom_max_iter(self):
        """Test array with custom max_iter value."""
        x0 = np.array([1.0, 2.0, 3.0])
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=x0, max_iter=75
        )
        
        assert np.all(iters == 75), "All iterations should use custom max_iter"
    
    def test_array_single_element(self):
        """Test array with single element (edge case between scalar and array)."""
        x0 = np.array([5.0])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (1,), "Should return 1-element array"
        assert iters.shape == (1,)
        assert conv.shape == (1,)
        
        assert np.isnan(roots[0])
        assert iters[0] == 100
        assert conv[0] is np.False_
    
    def test_array_very_large(self):
        """Test very large array (performance check)."""
        x0 = np.arange(100_000)
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (100_000,)
        assert np.all(np.isnan(roots))
        assert np.all(iters == 100)
        assert np.all(conv == False)
    
    def test_array_with_negative_values(self):
        """Test array with negative values."""
        x0 = np.array([-5.0, -2.0, -1.0])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (3,)
        assert np.all(np.isnan(roots))
    
    def test_array_with_mixed_values(self):
        """Test array with mixed positive/negative/zero values."""
        x0 = np.array([-10.0, 0.0, 5.0, 100.0])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (4,)
        assert np.all(np.isnan(roots))
        assert np.all(iters == 100)


class TestCreateSubstituteResultsPriority:
    """Test priority order: x0 > a > b."""
    
    def test_priority_x0_over_a(self):
        """Test that x0 takes priority over a."""
        x0 = np.array([1.0, 2.0])  # Length 2
        a = np.array([1.0, 2.0, 3.0])  # Length 3
        
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0, a=a)
        
        # Should use x0's length (2), not a's length (3)
        assert roots.shape == (2,), "Should use x0 length, not a length"
        assert iters.shape == (2,)
        assert conv.shape == (2,)
    
    def test_priority_x0_over_b(self):
        """Test that x0 takes priority over b."""
        x0 = np.array([1.0, 2.0, 3.0])  # Length 3
        b = np.array([1.0, 2.0])  # Length 2
        
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0, b=b)
        
        # Should use x0's length (3), not b's length (2)
        assert roots.shape == (3,)
    
    def test_priority_x0_over_both(self):
        """Test that x0 takes priority when all three provided."""
        x0 = np.array([1.0])  # Length 1
        a = np.array([1.0, 2.0])  # Length 2
        b = np.array([1.0, 2.0, 3.0])  # Length 3
        
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=x0, a=a, b=b
        )
        
        # Should use x0's length (1)
        assert roots.shape == (1,)
    
    def test_priority_a_over_b(self):
        """Test that a takes priority over b when x0 is None."""
        a = np.array([1.0, 2.0])  # Length 2
        b = np.array([1.0, 2.0, 3.0, 4.0])  # Length 4
        
        roots, iters, conv = RootSolvers._create_substitute_results(a=a, b=b)
        
        # Should use a's length (2), not b's length (4)
        assert roots.shape == (2,), "Should use a length, not b length"
    
    def test_priority_scalar_x0_over_array_a(self):
        """Test that scalar x0 takes priority over array a."""
        x0 = 5.0  # Scalar
        a = np.array([1.0, 2.0, 3.0])  # Array
        
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0, a=a)
        
        # Should return scalar (from x0), not array
        assert isinstance(roots, float), "Should return scalar from x0"
        assert np.isnan(roots)
    
    def test_priority_array_x0_over_scalar_b(self):
        """Test that array x0 takes priority over scalar b."""
        x0 = np.array([1.0, 2.0])  # Array
        b = 5.0  # Scalar
        
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0, b=b)
        
        # Should return array (from x0), not scalar
        assert isinstance(roots, np.ndarray), "Should return array from x0"
        assert roots.shape == (2,)


class TestCreateSubstituteResultsInputTypes:
    """Test different input types (list, tuple, ndarray, scalar)."""
    
    def test_list_input(self):
        """Test Python list as input."""
        x0 = [1.0, 2.0, 3.0]
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (3,)
        assert isinstance(roots, np.ndarray)
        assert np.all(np.isnan(roots))
    
    def test_tuple_input(self):
        """Test Python tuple as input."""
        x0 = (1.0, 2.0, 3.0, 4.0)
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (4,)
        assert isinstance(roots, np.ndarray)
    
    def test_nested_list_input(self):
        """Test nested list (2D array) - uses length of outer list."""
        x0 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        # np.asarray of 2D list is 2D array, len() gives first dimension
        assert roots.shape == (3,)
    
    def test_integer_list_input(self):
        """Test list of integers."""
        x0 = [1, 2, 3]
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (3,)
        assert roots.dtype == np.float64
    
    def test_numpy_int_array_input(self):
        """Test numpy array of integers."""
        x0 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (5,)
        assert roots.dtype == np.float64  # Output should be float64


class TestCreateSubstituteResultsEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_array(self):
        """Test empty array input."""
        x0 = np.array([])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (0,), "Should handle empty array"
        assert iters.shape == (0,)
        assert conv.shape == (0,)
        
        # Check dtypes even for empty arrays
        assert roots.dtype == np.float64
        assert iters.dtype == np.int64
        assert conv.dtype == bool
    
    def test_zero_dimensional_array(self):
        """Test 0-D numpy array (scalar wrapped in array)."""
        x0 = np.array(5.0)  # 0-D array
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        # Should treat as scalar
        assert isinstance(roots, float), "0-D array should return scalar"
        assert np.isnan(roots)
        assert iters == 100
        assert conv is False
    
    def test_multidimensional_array_2d(self):
        """Test 2D array (uses length of first dimension)."""
        x0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        # len() of 2D array gives first dimension (3)
        assert roots.shape == (3,)
    
    def test_nan_input_value(self):
        """Test NaN as input value (should still work)."""
        x0 = np.array([np.nan, 1.0, 2.0])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        # Should create results regardless of input values
        assert roots.shape == (3,)
        assert np.all(np.isnan(roots))
    
    def test_inf_input_value(self):
        """Test infinity as input value (should still work)."""
        x0 = np.array([np.inf, -np.inf, 1.0])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (3,)
        assert np.all(np.isnan(roots))
    
    def test_very_large_max_iter(self):
        """Test very large max_iter value."""
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=np.array([1.0, 2.0]),
            max_iter=1_000_000
        )
        
        assert np.all(iters == 1_000_000)
    
    def test_multiple_none_then_valid(self):
        """Test that None values are skipped correctly."""
        # x0=None, a=None, b=valid should work
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=None, a=None, b=np.array([1.0, 2.0])
        )
        
        assert roots.shape == (2,)


class TestCreateSubstituteResultsErrors:
    """Test error handling."""
    
    def test_all_none_raises_error(self):
        """Test that all None inputs raises ValueError."""
        with pytest.raises(ValueError, match="Cannot determine problem size"):
            RootSolvers._create_substitute_results(x0=None, a=None, b=None)
    
    def test_all_none_with_max_iter_raises_error(self):
        """Test that max_iter alone is not sufficient."""
        with pytest.raises(ValueError, match="Cannot determine problem size"):
            RootSolvers._create_substitute_results(max_iter=50)
    
    def test_error_message_content(self):
        """Test that error message is informative."""
        with pytest.raises(ValueError) as exc_info:
            RootSolvers._create_substitute_results()
        
        error_msg = str(exc_info.value)
        assert "x0" in error_msg, "Error should mention x0"
        assert "a" in error_msg, "Error should mention a"
        assert "b" in error_msg, "Error should mention b"
        assert "backup solvers" in error_msg.lower()


class TestCreateSubstituteResultsDataTypes:
    """Test output data types are correct."""
    
    def test_scalar_output_types(self):
        """Test scalar output types are exactly correct."""
        roots, iters, conv = RootSolvers._create_substitute_results(x0=1.0)
        
        # Python native types for scalars
        assert type(roots) is float
        assert type(iters) is int
        assert type(conv) is bool
    
    def test_array_output_dtypes(self):
        """Test array output dtypes are exactly correct."""
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=np.array([1.0, 2.0])
        )
        
        # NumPy array dtypes
        assert roots.dtype == np.float64, f"Expected float64, got {roots.dtype}"
        assert iters.dtype == np.int64, f"Expected int64, got {iters.dtype}"
        assert conv.dtype == bool, f"Expected bool, got {conv.dtype}"
    
    def test_array_not_object_dtype(self):
        """Test that arrays are not object dtype."""
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=np.array([1.0, 2.0, 3.0])
        )
        
        assert roots.dtype != object
        assert iters.dtype != object
        assert conv.dtype != object


class TestCreateSubstituteResultsIntegration:
    """Test integration scenarios matching real usage."""
    
    def test_usage_after_newton_failure_scalar(self):
        """Test typical usage after Newton solver fails (scalar)."""
        # Simulating: Newton failed, need substitute results
        x0 = 2.5
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=x0, max_iter=100
        )
        
        assert np.isnan(roots), "Should create NaN for failed Newton"
        assert iters == 100
        assert conv is False
    
    def test_usage_after_brent_failure_array(self):
        """Test typical usage after Brent solver fails (array)."""
        # Simulating: Brent failed on vectorized problem
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([5.0, 6.0, 7.0])
        
        roots, iters, conv = RootSolvers._create_substitute_results(
            a=a, b=b, max_iter=100
        )
        
        assert roots.shape == (3,)
        assert np.all(np.isnan(roots))
        assert np.all(conv == False), "All should be unconverged for backup chain"
    
    def test_usage_hybrid_solver_failure(self):
        """Test usage when hybrid solver has both x0 and brackets."""
        # Hybrid solver might have both
        x0 = np.array([1.5, 2.5, 3.5])
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=x0, a=a, b=b, max_iter=50
        )
        
        # Should use x0's shape (priority)
        assert roots.shape == (3,)
        assert iters[0] == 50
    
    def test_results_compatible_with_backup_chain(self):
        """Test that results are compatible with backup solver chain."""
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        # These results should be processable by backup solvers
        unconverged_idx = np.where(~conv)[0]
        
        assert len(unconverged_idx) == 5, "All should be unconverged"
        assert np.array_equal(unconverged_idx, np.arange(5))
        
        # Check that arrays are writable (for in-place updates)
        assert roots.flags['WRITEABLE']
        assert iters.flags['WRITEABLE']
        assert conv.flags['WRITEABLE']


class TestCreateSubstituteResultsConsistency:
    """Test consistency across multiple calls."""
    
    def test_deterministic_scalar(self):
        """Test that function is deterministic for scalar inputs."""
        results1 = RootSolvers._create_substitute_results(x0=5.0)
        results2 = RootSolvers._create_substitute_results(x0=5.0)
        
        assert np.isnan(results1[0]) and np.isnan(results2[0])
        assert results1[1] == results2[1]
        assert results1[2] == results2[2]
    
    def test_deterministic_array(self):
        """Test that function is deterministic for array inputs."""
        x0 = np.array([1.0, 2.0, 3.0])
        
        results1 = RootSolvers._create_substitute_results(x0=x0)
        results2 = RootSolvers._create_substitute_results(x0=x0)
        
        # All NaN comparisons
        assert_array_equal(
            np.isnan(results1[0]),
            np.isnan(results2[0])
        )
        assert_array_equal(results1[1], results2[1])
        assert_array_equal(results1[2], results2[2])
    
    def test_shape_consistency(self):
        """Test that output shapes always match input shapes."""
        test_cases = [
            (np.array([1.0]), (1,)),
            (np.array([1.0, 2.0]), (2,)),
            (np.arange(10), (10,)),
            (np.arange(100), (100,)),
        ]
        
        for x0, expected_shape in test_cases:
            roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
            assert roots.shape == expected_shape
            assert iters.shape == expected_shape
            assert conv.shape == expected_shape


# Parametrized tests for comprehensive coverage
class TestCreateSubstituteResultsParametrized:
    """Parametrized tests for systematic coverage."""
    
    @pytest.mark.parametrize("size", [1, 2, 5, 10, 50, 100, 1000])
    def test_various_array_sizes(self, size):
        """Test various array sizes systematically."""
        x0 = np.arange(size, dtype=float)
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        assert roots.shape == (size,)
        assert iters.shape == (size,)
        assert conv.shape == (size,)
        assert np.all(np.isnan(roots))
        assert np.all(iters == 100)
        assert np.all(conv == False)
    
    @pytest.mark.parametrize("max_iter_value", [0, 1, 10, 50, 100, 1000, 10000])
    def test_various_max_iter_values(self, max_iter_value):
        """Test various max_iter values."""
        roots, iters, conv = RootSolvers._create_substitute_results(
            x0=np.array([1.0, 2.0]),
            max_iter=max_iter_value
        )
        
        assert np.all(iters == max_iter_value)
    
    @pytest.mark.parametrize("input_param", ["x0", "a", "b"])
    def test_each_parameter_individually(self, input_param):
        """Test each parameter (x0, a, b) individually."""
        kwargs = {input_param: np.array([1.0, 2.0, 3.0])}
        
        roots, iters, conv = RootSolvers._create_substitute_results(**kwargs)
        
        assert roots.shape == (3,)
        assert np.all(np.isnan(roots))
    
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_input_dtypes(self, dtype):
        """Test that function works with various input dtypes."""
        x0 = np.array([1, 2, 3], dtype=dtype)
        roots, iters, conv = RootSolvers._create_substitute_results(x0=x0)
        
        # Output should always be float64/int64/bool regardless of input
        assert roots.dtype == np.float64
        assert iters.dtype == np.int64
        assert conv.dtype == bool


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])