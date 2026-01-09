"""
Comprehensive unit tests for vectorised Brent solver with codegen integration.

Tests parameter handling, parallel execution, and validates against scipy.

Author: Cian Quezon
"""
import numpy as np
import pytest
from numba import njit
from scipy.optimize import brentq

from meteorological_equations.math.solvers._jit_solvers import _brent_vectorised


class TestBrentVectorised:
    """Test vectorised Brent solver with codegen parameter handling."""

    def test_no_parameters_multiple_solves(self):
        """Test multiple solves with no parameters (None)."""
        @njit
        def f(x):
            return x**2 - 4
        
        # 100 solves, no parameters (None)
        a = np.zeros(100)
        b = np.ones(100) * 5.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=None, tol=1e-8, max_iter=100
        )
        
        # Compare with scipy
        expected = brentq(lambda x: x**2 - 4, 0.0, 5.0)
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)
        assert np.allclose(roots, 2.0, atol=1e-8)

    def test_no_parameters_omitted(self):
        """Test multiple solves with no parameters (omitted argument)."""
        @njit
        def f(x):
            return x**2 - 4
        
        # 100 solves, no parameters (omitted)
        a = np.zeros(100)
        b = np.ones(100) * 5.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b  # func_params omitted entirely
        )
        
        # Compare with scipy
        expected = brentq(lambda x: x**2 - 4, 0.0, 5.0)
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)
        assert np.allclose(roots, 2.0, atol=1e-8)

    def test_single_parameter_multiple_solves(self):
        """Test multiple solves with single parameter each."""
        @njit
        def f(x, k):
            return x**3 - k
        
        # 50 solves, 1 parameter each
        k_values = np.array([8.0, 27.0, 64.0, 125.0, 216.0])
        a = np.zeros(5)
        b = np.ones(5) * 10.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-8, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([brentq(lambda x: x**3 - k, 0.0, 10.0) 
                            for k in k_values])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)
        assert np.allclose(roots, np.cbrt(k_values), atol=1e-8)

    def test_two_parameters_multiple_solves(self):
        """Test multiple solves with two parameters each."""
        @njit
        def f(x, a, b):
            return a * x**3 - b
        
        # 30 solves, 2 parameters each
        n = 30
        a_values = np.ones(n)
        b_values = np.linspace(8.0, 216.0, n)
        func_params = np.column_stack([a_values, b_values])
        
        a_bounds = np.zeros(n)
        b_bounds = np.cbrt(b_values) + 2.0
        
        roots, iters, converged = _brent_vectorised(
            f, a_bounds, b_bounds, func_params=func_params, tol=1e-8, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([
            brentq(lambda x: a_values[i] * x**3 - b_values[i], 
                   a_bounds[i], b_bounds[i])
            for i in range(n)
        ])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)
        assert np.allclose(roots, np.cbrt(b_values), atol=1e-8)

    def test_three_parameters_atmospheric(self):
        """Test atmospheric equation with three parameters."""
        @njit
        def f(x, a, b, c):
            return a * x**3 + b * x + c
        
        # Cubic equations with known roots in brackets
        params = np.array([
            [1.0, 0.0, -8.0],    # x³ - 8,  root at 2.0
            [1.0, 0.0, -27.0],   # x³ - 27, root at 3.0
            [1.0, 0.0, -64.0],   # x³ - 64, root at 4.0
            [1.0, 0.0, -125.0],  # x³ - 125, root at 5.0
        ])
        
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=params, tol=1e-8, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([
            brentq(lambda x: params[i,0]*x**3 + params[i,1]*x + params[i,2],
                   a[i], b[i])
            for i in range(len(params))
        ])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)
        assert np.allclose(roots, [2.0, 3.0, 4.0, 5.0], atol=1e-8)

    def test_four_parameters_complex_atmospheric(self):
        """Test with four parameters (pressure-dependent atmospheric equation)."""
        @njit
        def f(T, T_surf, Td, P, gamma):
            # Simplified atmospheric equation
            return T - Td - gamma * (T_surf - T) * (P / 101325.0)
        
        # Multiple atmospheric profiles
        n = 25
        T_surf = np.random.uniform(280, 310, n)
        Td = T_surf - np.random.uniform(5, 15, n)
        P = np.random.uniform(50000, 101325, n)
        gamma = np.random.uniform(0.15, 0.25, n)
        
        func_params = np.column_stack([T_surf, Td, P, gamma])
        
        a = np.full(n, 250.0)
        b = np.full(n, 320.0)
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=func_params, tol=1e-6, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([
            brentq(
                lambda T: (T - func_params[i,1] - 
                          func_params[i,3] * (func_params[i,0] - T) * 
                          (func_params[i,2] / 101325.0)),
                a[i], b[i]
            )
            for i in range(n)
        ])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-6)

    def test_five_parameters(self):
        """Test with five parameters."""
        @njit
        def f(x, p0, p1, p2, p3, p4):
            return p0*x**4 + p1*x**3 + p2*x**2 + p3*x + p4
        
        # Quartic equations with roots at x=1
        params = np.array([
            [1.0, 0.0, -5.0, 0.0, 4.0],    # x⁴ - 5x² + 4, roots at ±1, ±2
            [1.0, 0.0, -10.0, 0.0, 9.0],   # x⁴ - 10x² + 9, roots at ±1, ±3
            [1.0, 0.0, -13.0, 0.0, 12.0],  # x⁴ - 13x² + 12
        ])
        
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([1.5, 1.5, 1.5])
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=params, tol=1e-8, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([
            brentq(
                lambda x: (params[i,0]*x**4 + params[i,1]*x**3 + 
                          params[i,2]*x**2 + params[i,3]*x + params[i,4]),
                a[i], b[i]
            )
            for i in range(len(params))
        ])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)
        assert np.allclose(roots, 1.0, atol=1e-8)

    def test_large_scale_parallel(self):
        """Test large-scale parallel execution (performance test)."""
        @njit
        def f(x, k):
            return x**2 - k
        
        # Large scale: 10,000 solves
        n = 10000
        k_values = np.linspace(1.0, 100.0, n)
        
        a = np.zeros(n)
        b = np.sqrt(k_values) + 2.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-8, max_iter=100
        )
        
        # All should converge
        assert np.all(converged)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-8)
        
        # Brent should converge efficiently
        assert np.mean(iters) < 15

    def test_mixed_convergence(self):
        """Test scenario with some converging and some failing."""
        @njit
        def f(x, offset):
            return x**2 + offset
        
        # Mix of valid (negative offset) and invalid (positive offset) cases
        offsets = np.array([-4.0, 1.0, -9.0, 5.0, -16.0, -25.0])
        
        a = np.zeros(6)
        b = np.ones(6) * 6.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=offsets, tol=1e-8, max_iter=100
        )
        
        # Check expected convergence pattern
        assert converged[0] == True   # offset = -4 (has root)
        assert converged[1] == False  # offset = 1 (no root)
        assert converged[2] == True   # offset = -9 (has root)
        assert converged[3] == False  # offset = 5 (no root)
        assert converged[4] == True   # offset = -16 (has root)
        assert converged[5] == True   # offset = -25 (has root)
        
        # Check converged roots
        assert np.isclose(roots[0], 2.0, atol=1e-8)
        assert np.isnan(roots[1])
        assert np.isclose(roots[2], 3.0, atol=1e-8)
        assert np.isnan(roots[3])
        assert np.isclose(roots[4], 4.0, atol=1e-8)
        assert np.isclose(roots[5], 5.0, atol=1e-8)

    def test_different_brackets_per_solve(self):
        """Test with different bracket intervals per solve."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k_values = np.array([8.0, 27.0, 64.0, 125.0])
        
        # Different brackets for each
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 6.0])
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-8, max_iter=100
        )
        
        expected = np.cbrt(k_values)
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)

    def test_very_tight_tolerance(self):
        """Test with extremely tight tolerance."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k_values = np.array([8.0, 27.0, 64.0])
        
        a = np.zeros(3)
        b = np.ones(3) * 5.0
        
        # Very tight tolerance
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-12, max_iter=100
        )
        
        assert np.all(converged)
        assert np.allclose(roots, np.cbrt(k_values), atol=1e-12)

    def test_reshape_single_parameter_array(self):
        """Test automatic reshaping of 1D parameter array."""
        @njit
        def f(x, k):
            return x**3 - k
        
        # 1D array (should be reshaped to (5, 1))
        k_values = np.array([8.0, 27.0, 64.0, 125.0, 216.0])
        
        a = np.zeros(5)
        b = np.ones(5) * 10.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-8, max_iter=100
        )
        
        assert np.all(converged)
        assert np.allclose(roots, np.cbrt(k_values), atol=1e-8)

    def test_transcendental_with_parameters(self):
        """Test transcendental equation with parameters."""
        @njit
        def f(x, a, b):
            return a * np.exp(x) - b
        
        a_values = np.ones(10)
        b_values = np.linspace(5.0, 50.0, 10)
        func_params = np.column_stack([a_values, b_values])
        
        a = np.zeros(10)
        b = np.ones(10) * 5.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=func_params, tol=1e-10, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([
            brentq(lambda x: a_values[i] * np.exp(x) - b_values[i], 0.0, 5.0)
            for i in range(10)
        ])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-10)
        # Should be ln(b/a)
        assert np.allclose(roots, np.log(b_values / a_values), atol=1e-10)

    def test_trigonometric_with_parameters(self):
        """Test trigonometric equation with parameters."""
        @njit
        def f(x, amplitude, offset):
            return amplitude * np.sin(x) - offset
        
        amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        offsets = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        func_params = np.column_stack([amplitudes, offsets])
        
        a = np.zeros(5)
        b = np.ones(5) * np.pi / 2
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=func_params, tol=1e-10, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([
            brentq(lambda x: amplitudes[i] * np.sin(x) - offsets[i], 0.0, np.pi/2)
            for i in range(5)
        ])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-10)

    def test_root_at_boundary(self):
        """Test when root is exactly at bracket boundary."""
        @njit
        def f(x, k):
            return x - k
        
        k_values = np.array([0.0, 5.0, 10.0])
        
        # Roots at lower boundary
        a = k_values
        b = k_values + 5.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-8, max_iter=100
        )
        
        assert np.all(converged)
        assert np.allclose(roots, k_values, atol=1e-8)

    def test_narrow_brackets(self):
        """Test with very narrow initial brackets."""
        @njit
        def f(x, k):
            return x**2 - k
        
        k_values = np.array([4.0, 9.0, 16.0])
        expected_roots = np.sqrt(k_values)
        
        # Narrow brackets around solution
        a = expected_roots - 0.1
        b = expected_roots + 0.1
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-8, max_iter=100
        )
        
        assert np.all(converged)
        assert np.allclose(roots, expected_roots, atol=1e-8)
        # Should converge very quickly with narrow bracket
        assert np.all(iters < 10)


class TestBrentConsistencyWithScipy:
    """Verify vectorised Brent matches scipy across various scenarios."""

    @pytest.mark.parametrize("n_solves", [10, 100, 1000])
    def test_consistency_various_sizes(self, n_solves):
        """Test consistency with scipy for different problem sizes."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k_values = np.random.uniform(1.0, 100.0, n_solves)
        
        a = np.zeros(n_solves)
        b = np.cbrt(k_values) + 2.0
        
        roots, _, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-10, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([brentq(lambda x: x**3 - k, 0.0, b[i]) 
                            for i, k in enumerate(k_values)])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-10)

    def test_davies_jones_wetbulb_scenario(self):
        """Test realistic Davies-Jones wetbulb calculation scenario."""
        @njit
        def simplified_wetbulb(Tw, T, Td, P):
            """Simplified wetbulb equation for testing."""
            return (Tw - Td) - 0.15 * (T - Tw) * (P / 101325.0)
        
        # 100 weather stations
        n = 100
        T = np.random.uniform(280, 310, n)
        Td = T - np.random.uniform(2, 15, n)
        P = np.random.uniform(70000, 101325, n)
        
        func_params = np.column_stack([T, Td, P])
        
        # Wetbulb must be between dewpoint and temperature
        a = Td
        b = T
        
        roots, iters, converged = _brent_vectorised(
            simplified_wetbulb, a, b, func_params=func_params, tol=1e-8, max_iter=100
        )
        
        # Compare with scipy
        expected = np.array([
            brentq(
                lambda Tw: (Tw - func_params[i,1]) - 
                          0.15 * (func_params[i,0] - Tw) * (func_params[i,2] / 101325.0),
                a[i], b[i]
            )
            for i in range(n)
        ])
        
        assert np.all(converged)
        assert np.allclose(roots, expected, atol=1e-8)
        
        # Physical constraints
        assert np.all(roots >= Td)
        assert np.all(roots <= T)

    def test_iteration_efficiency_vs_bisection(self):
        """Test that Brent is more efficient than bisection."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k_values = np.array([8.0, 27.0, 64.0, 125.0, 216.0])
        
        a = np.zeros(5)
        b = np.ones(5) * 10.0
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-10, max_iter=100
        )
        
        assert np.all(converged)
        # Brent should converge in fewer iterations than bisection
        # Bisection would need ~log2((b-a)/tol) ≈ 33 iterations
        # Brent should do better
        assert np.all(iters < 20)
        assert np.mean(iters) < 12


class TestBrentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_wide_brackets(self):
        """Test with very wide bracket intervals."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k = 8.0
        func_params = np.array([k])
        
        # Very wide bracket
        a = np.array([-1000.0])
        b = np.array([1000.0])
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=func_params, tol=1e-8, max_iter=100
        )
        
        assert converged[0]
        assert np.isclose(roots[0], 2.0, atol=1e-8)

    def test_zero_crossing_at_origin(self):
        """Test function that crosses zero at origin."""
        @njit
        def f(x, slope):
            return slope * x
        
        slopes = np.array([1.0, 2.0, 0.5, 3.0])
        
        a = np.full(4, -1.0)
        b = np.full(4, 1.0)
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=slopes, tol=1e-10, max_iter=100
        )
        
        assert np.all(converged)
        assert np.allclose(roots, 0.0, atol=1e-10)

    def test_multiple_roots_in_bracket(self):
        """Test function with multiple roots (Brent finds one)."""
        @njit
        def f(x, k):
            # Has roots at ±sqrt(k)
            return x**2 - k
        
        k = 4.0
        func_params = np.array([k])
        
        # Bracket contains positive root only
        a = np.array([0.0])
        b = np.array([5.0])
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=func_params, tol=1e-8, max_iter=100
        )
        
        assert converged[0]
        assert np.isclose(roots[0], 2.0, atol=1e-8)

    def test_flat_region(self):
        """Test function with nearly flat region."""
        @njit
        def f(x, k):
            # Nearly flat near x=0
            return x**5 - k
        
        k_values = np.array([0.00001, 0.0001, 0.001])
        
        a = np.zeros(3)
        b = np.ones(3) + 0.1
        
        roots, iters, converged = _brent_vectorised(
            f, a, b, func_params=k_values, tol=1e-8, max_iter=100
        )
        
        assert np.all(converged)
        assert np.allclose(roots, k_values**(1/5), atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])