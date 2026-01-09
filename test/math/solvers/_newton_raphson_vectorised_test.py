"""
Comprehensive unit tests for vectorised Newton-Raphson solver with codegen integration.

Tests parameter handling, parallel execution, and validates against scipy.

Author: Cian Quezon
"""

import numpy as np
import pytest
from numba import njit
from scipy.optimize import newton

from meteorological_equations.math.solvers._jit_solvers import _newton_raphson_vectorised


class TestNewtonRaphsonVectorised:
    """Test vectorised Newton-Raphson solver with codegen parameter handling."""

    def test_no_parameters_multiple_solves(self):
        """Test multiple solves with no parameters (None)."""

        @njit
        def f(x):
            return x**2 - 4

        @njit
        def fp(x):
            return 2 * x

        # 100 solves, no parameters (None)
        x0 = np.ones(100) * 1.5

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=None, tol=1e-6, max_iter=50)

        # Compare with scipy
        expected = np.array(
            [newton(lambda x: x**2 - 4, 1.5, fprime=lambda x: 2 * x) for _ in range(100)]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)
        assert np.allclose(roots, 2.0, atol=1e-6)

    def test_no_parameters_omitted(self):
        """Test multiple solves with no parameters (omitted)."""

        @njit
        def f(x):
            return x**2 - 4

        @njit
        def fp(x):
            return 2 * x

        # 100 solves, no parameters (omitted)
        x0 = np.ones(100) * 1.5

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0)  # func_params omitted

        # Compare with scipy
        expected = np.array(
            [newton(lambda x: x**2 - 4, 1.5, fprime=lambda x: 2 * x) for _ in range(100)]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)
        assert np.allclose(roots, 2.0, atol=1e-6)

    def test_single_parameter_multiple_solves(self):
        """Test multiple solves with single parameter each."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, k):
            return 2 * x

        # 50 solves, 1 parameter each
        k_values = np.linspace(4.0, 100.0, 50)
        x0 = np.sqrt(k_values) * 0.8  # Start near solution

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=k_values, tol=1e-6, max_iter=50)

        # Compare with scipy
        expected = np.array(
            [
                newton(lambda x: x**2 - k, x0[i], fprime=lambda x: 2 * x)
                for i, k in enumerate(k_values)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-6)

    def test_two_parameters_multiple_solves(self):
        """Test multiple solves with two parameters each."""

        @njit
        def f(x, a, b):
            return a * x**2 - b

        @njit
        def fp(x, a, b):
            return 2 * a * x

        # 30 solves, 2 parameters each
        a_values = np.ones(30)
        b_values = np.linspace(4.0, 100.0, 30)
        func_params = np.column_stack([a_values, b_values])
        x0 = np.sqrt(b_values) * 0.8

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=func_params, tol=1e-6, max_iter=50)

        # Compare with scipy
        expected = np.array(
            [
                newton(
                    lambda x: a_values[i] * x**2 - b_values[i],
                    x0[i],
                    fprime=lambda x: 2 * a_values[i] * x,
                )
                for i in range(len(a_values))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_three_parameters_atmospheric(self):
        """Test atmospheric equation with three parameters."""

        @njit
        def f(T, T_surf, Td, factor):
            return T - Td - factor * (T_surf - T)

        @njit
        def fp(T, T_surf, Td, factor):
            return 1.0 + factor

        # 100 weather stations, 3 parameters each
        n = 100
        T_surf = np.random.uniform(273, 310, n)
        Td = T_surf - np.random.uniform(5, 20, n)
        factor = np.random.uniform(0.1, 0.3, n)

        func_params = np.column_stack([T_surf, Td, factor])
        x0 = (T_surf + Td) / 2

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=func_params, tol=1e-6, max_iter=50)

        # Compare with scipy
        expected = np.array(
            [
                newton(
                    lambda T: T - func_params[i, 1] - func_params[i, 2] * (func_params[i, 0] - T),
                    x0[i],
                    fprime=lambda T: 1.0 + func_params[i, 2],
                )
                for i in range(n)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_four_parameters_complex_atmospheric(self):
        """Test with four parameters (pressure-dependent atmospheric equation)."""

        @njit
        def f(T, T_surf, Td, P, gamma):
            # Simplified atmospheric lapse rate equation
            return T - Td - gamma * (T_surf - T) * (P / 101325.0)

        @njit
        def fp(T, T_surf, Td, P, gamma):
            return 1.0 + gamma * (P / 101325.0)

        # Multiple atmospheric profiles
        n = 50
        T_surf = np.random.uniform(280, 310, n)
        Td = T_surf - np.random.uniform(5, 15, n)
        P = np.random.uniform(50000, 101325, n)
        gamma = np.random.uniform(0.15, 0.25, n)

        func_params = np.column_stack([T_surf, Td, P, gamma])
        x0 = Td + 5.0

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=func_params, tol=1e-6, max_iter=50)

        # Compare with scipy
        expected = np.array(
            [
                newton(
                    lambda T: (
                        T
                        - func_params[i, 1]
                        - func_params[i, 3]
                        * (func_params[i, 0] - T)
                        * (func_params[i, 2] / 101325.0)
                    ),
                    x0[i],
                    fprime=lambda T: 1.0 + func_params[i, 3] * (func_params[i, 2] / 101325.0),
                )
                for i in range(n)
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
        def fp(x, p0, p1, p2, p3, p4):
            return 4 * p0 * x**3 + 3 * p1 * x**2 + 2 * p2 * x + p3

        # Multiple polynomial solves
        params = np.array(
            [
                [1.0, 0.0, -10.0, 0.0, 9.0],  # x^4 - 10x^2 + 9
                [1.0, 0.0, -5.0, 0.0, 4.0],  # x^4 - 5x^2 + 4
                [1.0, 0.0, -13.0, 0.0, 12.0],  # x^4 - 13x^2 + 12
            ]
        )
        x0 = np.array([1.5, 1.3, 1.8])

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=params, tol=1e-6, max_iter=50)

        # Compare with scipy
        expected = np.array(
            [
                newton(
                    lambda x: (
                        params[i, 0] * x**4
                        + params[i, 1] * x**3
                        + params[i, 2] * x**2
                        + params[i, 3] * x
                        + params[i, 4]
                    ),
                    x0[i],
                    fprime=lambda x: (
                        4 * params[i, 0] * x**3
                        + 3 * params[i, 1] * x**2
                        + 2 * params[i, 2] * x
                        + params[i, 3]
                    ),
                )
                for i in range(len(params))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_large_scale_parallel(self):
        """Test large-scale parallel execution (performance test)."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, k):
            return 2 * x

        # Large scale: 10,000 solves
        n = 10000
        k_values = np.linspace(1.0, 100.0, n)
        x0 = np.sqrt(k_values) * 0.7

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=k_values, tol=1e-6, max_iter=50)

        # All should converge
        assert np.all(converged)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-6)

        # Most should converge quickly (< 10 iterations)
        assert np.mean(iters) < 10

    def test_mixed_convergence(self):
        """Test scenario with some converging and some failing."""

        @njit
        def f(x, offset):
            return x**2 + offset

        @njit
        def fp(x, offset):
            return 2 * x

        # Mix of valid (negative offset) and invalid (positive offset) cases
        offsets = np.array([-4.0, 1.0, -9.0, 5.0, -16.0])
        x0 = np.ones(5) * 1.0

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=offsets, tol=1e-6, max_iter=50)

        # Check expected convergence pattern
        assert converged[0] == True  # offset = -4
        assert converged[1] == False  # offset = 1 (no real roots)
        assert converged[2] == True  # offset = -9
        assert converged[3] == False  # offset = 5 (no real roots)
        assert converged[4] == True  # offset = -16

        # Check converged roots are correct
        assert np.isclose(roots[0], 2.0, atol=1e-6)
        assert np.isclose(roots[2], 3.0, atol=1e-6)
        assert np.isclose(roots[4], 4.0, atol=1e-6)

    def test_different_starting_points(self):
        """Test same equation with different starting points."""

        @njit
        def f(x, k):
            return x**3 - k

        @njit
        def fp(x, k):
            return 3 * x**2

        # Same parameter, different starting points
        k = 27.0
        func_params = np.full(5, k)
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=func_params, tol=1e-6, max_iter=50)

        # All should converge to same root (3.0)
        assert np.all(converged)
        assert np.allclose(roots, 3.0, atol=1e-6)

    def test_tight_tolerance(self):
        """Test with very tight tolerance."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, k):
            return 2 * x

        k_values = np.array([4.0, 9.0, 16.0])
        x0 = np.ones(3) * 1.5

        # Very tight tolerance
        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=k_values, tol=1e-12, max_iter=100)

        assert np.all(converged)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-12)

    def test_reshape_single_parameter_array(self):
        """Test automatic reshaping of 1D parameter array."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, k):
            return 2 * x

        # 1D array (should be reshaped to (5, 1))
        k_values = np.array([4.0, 9.0, 16.0, 25.0, 36.0])
        x0 = np.ones(5) * 1.5

        roots, iters, converged = _newton_raphson_vectorised(f, fp, x0, func_params=k_values, tol=1e-6, max_iter=50)

        assert np.all(converged)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-6)


class TestVectorisedConsistencyWithScipy:
    """Verify vectorised solver matches scipy across various scenarios."""

    @pytest.mark.parametrize("n_solves", [10, 100, 1000])
    def test_consistency_various_sizes(self, n_solves):
        """Test consistency with scipy for different problem sizes."""

        @njit
        def f(x, k):
            return x**2 - k

        @njit
        def fp(x, k):
            return 2 * x

        k_values = np.random.uniform(1.0, 100.0, n_solves)
        x0 = np.sqrt(k_values) * 0.8

        roots, _, converged = _newton_raphson_vectorised(f, fp, x0, func_params=k_values, tol=1e-8, max_iter=50)

        # Compare with scipy
        expected = np.array(
            [
                newton(lambda x: x**2 - k, x0[i], fprime=lambda x: 2 * x)
                for i, k in enumerate(k_values)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)

    def test_davies_jones_wetbulb_scenario(self):
        """Test realistic Davies-Jones wetbulb calculation scenario."""

        @njit
        def simplified_wetbulb(Tw, T, Td, P):
            """Simplified wetbulb equation for testing."""
            # Simplified form - actual Davies-Jones is more complex
            return (Tw - Td) - 0.15 * (T - Tw) * (P / 101325.0)

        @njit
        def simplified_wetbulb_prime(Tw, T, Td, P):
            return 1.0 + 0.15 * (P / 101325.0)

        # 100 weather stations
        n = 100
        T = np.random.uniform(280, 310, n)
        Td = T - np.random.uniform(2, 15, n)
        P = np.random.uniform(70000, 101325, n)

        func_params = np.column_stack([T, Td, P])
        x0 = Td + 2.0  # Start slightly above dewpoint

        roots, iters, converged = _newton_raphson_vectorised(
            simplified_wetbulb, simplified_wetbulb_prime, x0, func_params=func_params, tol=1e-6, max_iter=50
        )

        # Compare with scipy
        expected = np.array(
            [
                newton(
                    lambda Tw: (Tw - func_params[i, 1])
                    - 0.15 * (func_params[i, 0] - Tw) * (func_params[i, 2] / 101325.0),
                    x0[i],
                    fprime=lambda Tw: 1.0 + 0.15 * (func_params[i, 2] / 101325.0),
                )
                for i in range(n)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

        # Physical constraints: Tw should be between Td and T
        assert np.all(roots >= Td)
        assert np.all(roots <= T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])