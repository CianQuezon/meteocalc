"""
Comprehensive unit tests for codegen vectorised solvers.

Author: Cian Quezon
"""

import numpy as np
import pytest
from numba import njit
from scipy.optimize import brentq, newton

from meteorological_equations.math.solvers._codegen import generate_vectorised_solver
from meteorological_equations.math.solvers._enums import MethodType
from meteorological_equations.math.solvers._jit_solvers import (
    _bisection_scalar,
    _brent_scalar,
    _newton_raphson_scalar,
)


class TestCodegenOpenMethods:
    """Test codegen for open methods (Newton-Raphson) with various parameters."""

    def test_no_parameters(self):
        """Test Newton-Raphson codegen with no parameters."""

        @njit
        def f(x):
            return x**2 - 4

        @njit
        def fp(x):
            return 2 * x

        # Generate solver
        solver = generate_vectorised_solver(_newton_raphson_scalar, 0, MethodType.OPEN)

        # Test data
        func_params = np.empty((100, 0), dtype=np.float64)  # No parameters
        x0 = np.ones(100) * 1.5

        # Solve
        roots, iters, converged = solver(f, fp, func_params, x0, 1e-6, 50)

        # Compare with scipy
        expected = np.array(
            [newton(lambda x: x**2 - 4, 1.5, fprime=lambda x: 2 * x) for _ in range(100)]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)
        assert np.allclose(roots, 2.0, atol=1e-6)

    def test_single_parameter(self):
        """Test Newton-Raphson codegen with single parameter."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, _k):
            return 2 * x

        solver = generate_vectorised_solver(_newton_raphson_scalar, 1, MethodType.OPEN)

        # Different k values for each solve
        k_values = np.array([4.0, 9.0, 16.0, 25.0, 36.0])
        func_params = k_values.reshape(-1, 1)
        x0 = np.ones(5) * 1.5

        roots, iters, converged = solver(f, fp, func_params, x0, 1e-6, 50)

        # Compare with scipy
        expected = np.array(
            [newton(lambda x, k=k: x**2 - k, 1.5, fprime=lambda x: 2 * x) for k in k_values]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-6)

    def test_three_parameters_atmospheric(self):
        """Test with three parameters (typical atmospheric LCL-like equation)."""

        @njit
        def f(T, T_surf, Td, factor):
            return T - Td - factor * (T_surf - T)

        @njit
        def fp(_T, _T_surf, _Td, factor):
            return 1.0 + factor

        solver = generate_vectorised_solver(_newton_raphson_scalar, 3, MethodType.OPEN)

        # Different atmospheric conditions
        params = np.array(
            [
                [293.15, 283.15, 0.2],  # 20°C, 10°C dewpoint
                [303.15, 298.15, 0.15],  # 30°C, 25°C dewpoint
                [273.15, 268.15, 0.25],  # 0°C, -5°C dewpoint
                [288.15, 278.15, 0.18],  # 15°C, 5°C dewpoint
                [298.15, 288.15, 0.22],  # 25°C, 15°C dewpoint
            ]
        )
        x0 = np.array([285.0, 300.0, 270.0, 280.0, 290.0])

        roots, iters, converged = solver(f, fp, params, x0, 1e-6, 50)

        # Compare with scipy
        expected = np.array(
            [
                newton(
                    lambda T, i=i: T - params[i, 1] - params[i, 2] * (params[i, 0] - T),
                    x0[i],
                    fprime=lambda _T, i=i: 1.0 + params[i, 2],
                )
                for i in range(len(params))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_five_parameters(self):
        """Test with five parameters."""

        @njit
        def f(x, p0, p1, p2, p3, p4):
            return p0 * x**4 + p1 * x**3 + p2 * x**2 + p3 * x + p4

        @njit
        def fp(x, p0, p1, p2, p3, _p4):
            return 4 * p0 * x**3 + 3 * p1 * x**2 + 2 * p2 * x + p3

        solver = generate_vectorised_solver(_newton_raphson_scalar, 5, MethodType.OPEN)

        # x^4 - 10x^2 + 9 has roots at ±1, ±3
        params = np.array(
            [
                [1.0, 0.0, -10.0, 0.0, 9.0],
                [1.0, 0.0, -10.0, 0.0, 9.0],
                [1.0, 0.0, -10.0, 0.0, 9.0],
            ]
        )
        x0 = np.array([0.5, 1.5, 2.5])

        roots, iters, converged = solver(f, fp, params, x0, 1e-6, 50)

        # Compare with scipy - whatever scipy gets, we should get
        expected = np.array(
            [
                newton(
                    lambda x, i=i: params[i, 0] * x**4
                    + params[i, 1] * x**3
                    + params[i, 2] * x**2
                    + params[i, 3] * x
                    + params[i, 4],
                    x0[i],
                    fprime=lambda x, i=i: 4 * params[i, 0] * x**3
                    + 3 * params[i, 1] * x**2
                    + 2 * params[i, 2] * x
                    + params[i, 3],
                )
                for i in range(len(params))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_large_array(self):
        """Test codegen with large array (performance test)."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, _k):
            return 2 * x

        solver = generate_vectorised_solver(_newton_raphson_scalar, 1, MethodType.OPEN)

        n = 10000
        k_values = np.linspace(1.0, 100.0, n)
        func_params = k_values.reshape(-1, 1)
        x0 = np.sqrt(k_values) * 0.8  # Start near expected roots

        roots, iters, converged = solver(f, fp, func_params, x0, 1e-6, 50)

        assert np.all(converged)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-6)


class TestCodegenBracketMethods:
    """Test codegen for bracket methods (Bisection, Brent) with various parameters."""

    def test_bisection_no_parameters(self):
        """Test bisection codegen with no parameters."""

        @njit
        def f(x):
            return x**2 - 4

        solver = generate_vectorised_solver(_bisection_scalar, 0, MethodType.BRACKET)

        func_params = np.empty((50, 0), dtype=np.float64)
        a = np.zeros(50)
        b = np.ones(50) * 5.0

        roots, iters, converged = solver(f, func_params, a, b, 1e-6, 100)

        expected = brentq(lambda x: x**2 - 4, 0.0, 5.0)

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_bisection_single_parameter(self):
        """Test bisection codegen with single parameter."""

        @njit
        def f(x, k):
            return x**3 - k

        solver = generate_vectorised_solver(_bisection_scalar, 1, MethodType.BRACKET)

        k_values = np.array([8.0, 27.0, 64.0, 125.0])
        func_params = k_values.reshape(-1, 1)
        a = np.zeros(4)
        b = np.ones(4) * 10.0

        roots, iters, converged = solver(f, func_params, a, b, 1e-6, 100)

        expected = np.array([brentq(lambda x, k=k: x**3 - k, 0.0, 10.0) for k in k_values])

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)
        assert np.allclose(roots, np.cbrt(k_values), atol=1e-6)

    def test_brent_three_parameters(self):
        """Test Brent codegen with three parameters."""

        @njit
        def f(x, a, b, c):
            return a * x**3 + b * x + c

        solver = generate_vectorised_solver(_brent_scalar, 3, MethodType.BRACKET)

        params = np.array(
            [
                [1.0, -2.0, -5.0],
                [1.0, -3.0, -8.0],
                [1.0, -1.0, -2.0],
            ]
        )

        # Use brackets that work for all three
        a_vals = np.array([1.5, 2.0, 1.0])
        b_vals = np.array([2.5, 3.0, 2.0])

        # Solve with your codegen solver
        roots, iters, converged = solver(f, params, a_vals, b_vals, 1e-8, 100)

        # Get expected from scipy with SAME inputs
        expected = np.array(
            [
                brentq(
                    lambda x, i=i: params[i, 0] * x**3 + params[i, 1] * x + params[i, 2],
                    a_vals[i],
                    b_vals[i],
                )
                for i in range(len(params))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)

    def test_bracket_different_intervals(self):
        """Test with different bracket intervals per solve."""

        @njit
        def f(x, offset):
            return x**2 + offset

        solver = generate_vectorised_solver(_bisection_scalar, 1, MethodType.BRACKET)

        offsets = np.array([-4.0, -9.0, -16.0, -25.0])
        func_params = offsets.reshape(-1, 1)

        # Different brackets for each
        a = np.array([0.0, 0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])

        roots, iters, converged = solver(f, func_params, a, b, 1e-6, 100)

        expected = np.sqrt(-offsets)

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_atmospheric_equation_four_params(self):
        """Test with realistic atmospheric equation (4 parameters)."""

        @njit
        def f(T, T_surf, Td, P, factor):
            return T - Td - factor * (T_surf - T) * (P / 101325.0)

        solver = generate_vectorised_solver(_brent_scalar, 4, MethodType.BRACKET)

        # Different atmospheric conditions at different pressure levels
        params = np.array(
            [
                [293.15, 283.15, 101325.0, 0.2],
                [293.15, 283.15, 85000.0, 0.2],
                [293.15, 283.15, 70000.0, 0.2],
            ]
        )

        a = np.ones(3) * 250.0
        b = np.ones(3) * 300.0

        roots, iters, converged = solver(f, params, a, b, 1e-4, 100)

        expected = np.array(
            [
                brentq(
                    lambda T, i=i: T
                    - params[i, 1]
                    - params[i, 3] * (params[i, 0] - T) * (params[i, 2] / 101325.0),
                    250.0,
                    300.0,
                )
                for i in range(len(params))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-4)


class TestCodegenCaching:
    """Test that codegen properly handles multiple calls and caching behavior."""

    def test_same_params_different_data(self):
        """Test same generated solver with different data."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, _k):
            return 2 * x

        solver = generate_vectorised_solver(_newton_raphson_scalar, 1, MethodType.OPEN)

        # First call
        params1 = np.array([[4.0], [9.0]])
        x0_1 = np.ones(2) * 1.5
        roots1, _, conv1 = solver(f, fp, params1, x0_1, 1e-6, 50)

        # Second call with different data
        params2 = np.array([[16.0], [25.0], [36.0]])
        x0_2 = np.ones(3) * 1.5
        roots2, _, conv2 = solver(f, fp, params2, x0_2, 1e-6, 50)

        assert np.all(conv1)
        assert np.all(conv2)
        assert np.allclose(roots1, [2.0, 3.0], atol=1e-6)
        assert np.allclose(roots2, [4.0, 5.0, 6.0], atol=1e-6)

    def test_different_param_counts(self):
        """Test generating solvers with different parameter counts."""

        @njit
        def f1(x, k):
            return x**2 - k

        @njit
        def f2(x, a, b):
            return a * x**2 - b

        @njit
        def fp1(x, _k):
            return 2 * x

        @njit
        def fp2(x, a, _b):
            return 2 * a * x

        # Generate two different solvers
        solver1 = generate_vectorised_solver(_newton_raphson_scalar, 1, MethodType.OPEN)
        solver2 = generate_vectorised_solver(_newton_raphson_scalar, 2, MethodType.OPEN)

        # Use them independently
        params1 = np.array([[4.0]])
        roots1, _, _ = solver1(f1, fp1, params1, np.array([1.5]), 1e-6, 50)

        params2 = np.array([[1.0, 4.0]])
        roots2, _, _ = solver2(f2, fp2, params2, np.array([1.5]), 1e-6, 50)

        assert np.isclose(roots1[0], 2.0, atol=1e-6)
        assert np.isclose(roots2[0], 2.0, atol=1e-6)


class TestCodegenEdgeCases:
    """Test edge cases and error conditions."""

    def test_mixed_convergence(self):
        """Test with some converging and some failing cases."""

        @njit
        def f(x, offset):
            return x**2 + offset

        solver = generate_vectorised_solver(_bisection_scalar, 1, MethodType.BRACKET)

        # offset=-4 converges, offset=1 fails (no real roots)
        offsets = np.array([[-4.0], [1.0], [-9.0]])
        a = np.zeros(3)
        b = np.ones(3) * 5.0

        roots, iters, converged = solver(f, offsets, a, b, 1e-6, 100)

        assert converged[0]
        assert not converged[1]
        assert converged[2]
        assert np.isnan(roots[1])

    def test_invalid_method_type(self):
        """Test error handling for invalid method type."""
        with pytest.raises(ValueError) as exc_info:
            generate_vectorised_solver(_newton_raphson_scalar, 1, "INVALID")

        assert "Invalid enum 'INVALID'" in str(exc_info.value)
        assert "Available are the following" in str(exc_info.value)

    def test_zero_parameters_bracket(self):
        """Test bracket method with zero parameters."""

        @njit
        def f(x):
            return x**3 - 8

        solver = generate_vectorised_solver(_brent_scalar, 0, MethodType.BRACKET)

        func_params = np.empty((3, 0), dtype=np.float64)
        a = np.zeros(3)
        b = np.ones(3) * 5.0

        roots, iters, converged = solver(f, func_params, a, b, 1e-8, 100)

        assert np.all(converged)
        assert np.allclose(roots, 2.0, atol=1e-8)


class TestCodegenPerformance:
    """Test performance characteristics of generated code."""

    def test_parallel_execution_large_scale(self):
        """Test that solver handles large-scale problems efficiently."""

        @njit
        def f(x, a, b):
            return a * x**2 - b

        @njit
        def fp(x, a, _b):
            return 2 * a * x

        solver = generate_vectorised_solver(_newton_raphson_scalar, 2, MethodType.OPEN)

        # Large problem size
        n = 100000
        params = np.column_stack([np.ones(n), np.linspace(1.0, 100.0, n)])
        x0 = np.ones(n) * 5.0

        roots, iters, converged = solver(f, fp, params, x0, 1e-6, 50)

        # Should converge for all
        assert np.sum(converged) / n > 0.99  # At least 99% convergence

        # Verify correctness on sample
        sample_indices = [0, n // 2, n - 1]
        for i in sample_indices:
            expected = np.sqrt(params[i, 1] / params[i, 0])
            assert np.isclose(roots[i], expected, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
