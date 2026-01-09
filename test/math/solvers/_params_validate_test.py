"""
Comprehensive unit tests for parameter validation helper function.

Tests all edge cases, input types, and error conditions.

Author: Cian Quezon
"""
import numpy as np
import pytest

from meteorological_equations.math.solvers._jit_solvers import _validate_and_prepare_params


class TestValidateAndPrepareParams:
    """Test parameter validation and preparation helper."""

    def test_none_parameters_small_scale(self):
        """Test with no parameters (None), small scale."""
        n_solves = 10
        
        params, num_params = _validate_and_prepare_params(None, n_solves)
        
        assert params.shape == (10, 0)
        assert params.dtype == np.float64
        assert num_params == 0

    def test_none_parameters_large_scale(self):
        """Test with no parameters (None), large scale."""
        n_solves = 100000
        
        params, num_params = _validate_and_prepare_params(None, n_solves)
        
        assert params.shape == (100000, 0)
        assert params.dtype == np.float64
        assert num_params == 0

    def test_single_parameter_1d_array(self):
        """Test single parameter as 1D array."""
        n_solves = 100
        func_params = np.random.uniform(1.0, 100.0, n_solves)
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (100, 1)
        assert params.dtype == np.float64
        assert num_params == 1
        
        # Verify data is preserved
        assert np.allclose(params[:, 0], func_params)

    def test_single_parameter_list_input(self):
        """Test single parameter as Python list."""
        n_solves = 50
        func_params = [1.0, 2.0, 3.0] * 16 + [4.0, 5.0]  # 50 elements
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (50, 1)
        assert params.dtype == np.float64
        assert num_params == 1

    def test_multiple_parameters_2d_array(self):
        """Test multiple parameters as 2D array."""
        n_solves = 100
        func_params = np.random.rand(n_solves, 3)
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (100, 3)
        assert params.dtype == np.float64
        assert num_params == 3
        
        # Verify data is preserved
        assert np.allclose(params, func_params)

    def test_large_parameter_count(self):
        """Test with many parameters (5 parameters)."""
        n_solves = 1000
        func_params = np.random.rand(n_solves, 5)
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (1000, 5)
        assert params.dtype == np.float64
        assert num_params == 5

    def test_single_solve_no_params(self):
        """Test edge case: single solve, no parameters."""
        params, num_params = _validate_and_prepare_params(None, 1)
        
        assert params.shape == (1, 0)
        assert num_params == 0

    def test_single_solve_single_param(self):
        """Test edge case: single solve, single parameter."""
        func_params = np.array([42.0])
        
        params, num_params = _validate_and_prepare_params(func_params, 1)
        
        assert params.shape == (1, 1)
        assert params[0, 0] == 42.0
        assert num_params == 1

    def test_single_solve_multiple_params(self):
        """Test edge case: single solve, multiple parameters."""
        func_params = np.array([[1.0, 2.0, 3.0]])
        
        params, num_params = _validate_and_prepare_params(func_params, 1)
        
        assert params.shape == (1, 3)
        assert np.allclose(params[0], [1.0, 2.0, 3.0])
        assert num_params == 3

    def test_integer_input_converted_to_float(self):
        """Test that integer arrays are converted to float64."""
        n_solves = 10
        func_params = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.dtype == np.float64
        assert params.shape == (10, 1)
        assert num_params == 1

    def test_mixed_numeric_types_2d(self):
        """Test 2D array with mixed numeric types."""
        n_solves = 5
        func_params = np.array([
            [1, 2.5, 3],
            [4, 5.5, 6],
            [7, 8.5, 9],
            [10, 11.5, 12],
            [13, 14.5, 15]
        ])
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.dtype == np.float64
        assert params.shape == (5, 3)
        assert num_params == 3

    def test_data_preservation_1d(self):
        """Test that 1D data is preserved after reshape."""
        n_solves = 100
        original = np.random.uniform(-100, 100, n_solves)
        
        params, _ = _validate_and_prepare_params(original.copy(), n_solves)
        
        assert np.allclose(params[:, 0], original)

    def test_data_preservation_2d(self):
        """Test that 2D data is preserved exactly."""
        n_solves = 50
        original = np.random.uniform(-100, 100, (n_solves, 4))
        
        params, _ = _validate_and_prepare_params(original.copy(), n_solves)
        
        assert np.allclose(params, original)
        assert params is not original  # Should be a copy


class TestValidateAndPrepareParamsErrors:
    """Test error conditions and validation."""

    def test_error_1d_length_too_short(self):
        """Test error when 1D array length is too short."""
        n_solves = 100
        func_params = np.ones(50)  # Only 50, need 100
        
        with pytest.raises(ValueError) as exc_info:
            _validate_and_prepare_params(func_params, n_solves)
        
        assert "length (50)" in str(exc_info.value)
        assert "number of solves (100)" in str(exc_info.value)

    def test_error_1d_length_too_long(self):
        """Test error when 1D array length is too long."""
        n_solves = 50
        func_params = np.ones(100)  # 100 elements, need 50
        
        with pytest.raises(ValueError) as exc_info:
            _validate_and_prepare_params(func_params, n_solves)
        
        assert "length (100)" in str(exc_info.value)
        assert "number of solves (50)" in str(exc_info.value)

    def test_error_2d_rows_too_few(self):
        """Test error when 2D array has too few rows."""
        n_solves = 100
        func_params = np.ones((50, 3))  # Only 50 rows, need 100
        
        with pytest.raises(ValueError) as exc_info:
            _validate_and_prepare_params(func_params, n_solves)
        
        assert "rows (50)" in str(exc_info.value)
        assert "number of solves (100)" in str(exc_info.value)

    def test_error_2d_rows_too_many(self):
        """Test error when 2D array has too many rows."""
        n_solves = 50
        func_params = np.ones((100, 3))  # 100 rows, need 50
        
        with pytest.raises(ValueError) as exc_info:
            _validate_and_prepare_params(func_params, n_solves)
        
        assert "rows (100)" in str(exc_info.value)
        assert "number of solves (50)" in str(exc_info.value)

    def test_error_message_clarity_1d(self):
        """Test that 1D error messages are clear and helpful."""
        with pytest.raises(ValueError) as exc_info:
            _validate_and_prepare_params(np.ones(10), 20)
        
        error_msg = str(exc_info.value)
        assert "func_params length" in error_msg
        assert "10" in error_msg
        assert "20" in error_msg
        assert "must match" in error_msg

    def test_error_message_clarity_2d(self):
        """Test that 2D error messages are clear and helpful."""
        with pytest.raises(ValueError) as exc_info:
            _validate_and_prepare_params(np.ones((10, 3)), 20)
        
        error_msg = str(exc_info.value)
        assert "func_params rows" in error_msg
        assert "10" in error_msg
        assert "20" in error_msg
        assert "must match" in error_msg


class TestValidateAndPrepareParamsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_params_zero_solves(self):
        """Test edge case: zero solves."""
        params, num_params = _validate_and_prepare_params(None, 0)
        
        assert params.shape == (0, 0)
        assert num_params == 0

    def test_very_large_parameter_array(self):
        """Test with very large parameter array."""
        n_solves = 1000000
        func_params = np.ones((n_solves, 2))
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (1000000, 2)
        assert num_params == 2

    def test_single_column_2d_array(self):
        """Test 2D array with single column (edge case)."""
        n_solves = 50
        func_params = np.ones((n_solves, 1))
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (50, 1)
        assert num_params == 1

    def test_wide_array_many_columns(self):
        """Test 2D array with many columns."""
        n_solves = 10
        func_params = np.ones((n_solves, 20))
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (10, 20)
        assert num_params == 20

    def test_special_values_nan(self):
        """Test that NaN values are preserved."""
        n_solves = 5
        func_params = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (5, 1)
        assert np.isnan(params[1, 0])
        assert np.isnan(params[3, 0])

    def test_special_values_inf(self):
        """Test that infinity values are preserved."""
        n_solves = 4
        func_params = np.array([1.0, np.inf, -np.inf, 4.0])
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (4, 1)
        assert np.isinf(params[1, 0]) and params[1, 0] > 0
        assert np.isinf(params[2, 0]) and params[2, 0] < 0

    def test_negative_values(self):
        """Test that negative values are handled correctly."""
        n_solves = 100
        func_params = np.random.uniform(-1000, 1000, (n_solves, 3))
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (100, 3)
        assert np.allclose(params, func_params)

    def test_zero_values(self):
        """Test that zero values are handled correctly."""
        n_solves = 50
        func_params = np.zeros((n_solves, 2))
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (50, 2)
        assert np.all(params == 0.0)


class TestValidateAndPrepareParamsIntegration:
    """Integration tests simulating real solver usage."""

    def test_atmospheric_wetbulb_scenario(self):
        """Test realistic atmospheric wetbulb calculation scenario."""
        n_stations = 100
        
        # Atmospheric parameters: T, Td, P
        T = np.random.uniform(280, 310, n_stations)
        Td = T - np.random.uniform(2, 15, n_stations)
        P = np.random.uniform(70000, 101325, n_stations)
        
        func_params = np.column_stack([T, Td, P])
        
        params, num_params = _validate_and_prepare_params(func_params, n_stations)
        
        assert params.shape == (100, 3)
        assert num_params == 3
        assert np.allclose(params[:, 0], T)
        assert np.allclose(params[:, 1], Td)
        assert np.allclose(params[:, 2], P)

    def test_simple_quadratic_scenario(self):
        """Test simple quadratic f(x) = x² - k scenario."""
        n_solves = 1000
        k_values = np.random.uniform(1.0, 100.0, n_solves)
        
        params, num_params = _validate_and_prepare_params(k_values, n_solves)
        
        assert params.shape == (1000, 1)
        assert num_params == 1
        assert np.allclose(params[:, 0], k_values)

    def test_no_parameter_function_scenario(self):
        """Test scenario with no parameters (f(x) = x² - 4)."""
        n_solves = 500
        
        params, num_params = _validate_and_prepare_params(None, n_solves)
        
        assert params.shape == (500, 0)
        assert num_params == 0

    def test_polynomial_coefficients_scenario(self):
        """Test polynomial with multiple coefficients."""
        n_solves = 200
        
        # f(x) = a*x³ + b*x² + c*x + d
        a = np.random.uniform(-5, 5, n_solves)
        b = np.random.uniform(-5, 5, n_solves)
        c = np.random.uniform(-5, 5, n_solves)
        d = np.random.uniform(-5, 5, n_solves)
        
        func_params = np.column_stack([a, b, c, d])
        
        params, num_params = _validate_and_prepare_params(func_params, n_solves)
        
        assert params.shape == (200, 4)
        assert num_params == 4

    def test_mixed_usage_pattern(self):
        """Test alternating between None and actual parameters."""
        n_solves = 100
        
        # First call: no parameters
        params1, num_params1 = _validate_and_prepare_params(None, n_solves)
        assert params1.shape == (100, 0)
        assert num_params1 == 0
        
        # Second call: single parameter
        single_param = np.ones(n_solves)
        params2, num_params2 = _validate_and_prepare_params(single_param, n_solves)
        assert params2.shape == (100, 1)
        assert num_params2 == 1
        
        # Third call: multiple parameters
        multi_param = np.ones((n_solves, 3))
        params3, num_params3 = _validate_and_prepare_params(multi_param, n_solves)
        assert params3.shape == (100, 3)
        assert num_params3 == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])