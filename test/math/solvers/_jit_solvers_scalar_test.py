"""
Comprehensive unit tests for scalar JIT-compiled root-finding solvers.

Tests parameter unpacking with *args and validates against scipy reference implementations.

Author: Cian Quezon
"""
import numpy as np
import pytest
from numba import njit
from scipy.optimize import newton, brentq

from meteorological_equations.math.solvers._jit_solvers import (
    _newton_raphson_scalar,
    _bisection_scalar,
    _brent_scalar,
)


class TestNewtonRaphsonScalar:
    """Test Newton-Raphson scalar solver with various parameter configurations."""

    def test_no_parameters(self):
        """Test with function requiring no parameters."""
        @njit
        def f(x):
            return x**2 - 4
        
        @njit
        def fp(x):
            return 2 * x
        
        root, iters, converged = _newton_raphson_scalar(f, fp, 1.0, 1e-6, 50)
        scipy_root = newton(lambda x: x**2 - 4, 1.0, fprime=lambda x: 2*x)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 2.0, atol=1e-6)

    def test_single_parameter(self):
        """Test with one parameter."""
        @njit
        def f(x, k):
            return x**2 - k
        
        @njit
        def fp(x, k):
            return 2 * x
        
        k = 9.0
        root, iters, converged = _newton_raphson_scalar(f, fp, 2.0, 1e-6, 50, k)
        scipy_root = newton(lambda x: x**2 - k, 2.0, fprime=lambda x: 2*x)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 3.0, atol=1e-6)

    def test_two_parameters(self):
        """Test with two parameters."""
        @njit
        def f(x, a, b):
            return a * x**2 - b
        
        @njit
        def fp(x, a, b):
            return 2 * a * x
        
        a, b = 2.0, 8.0
        root, iters, converged = _newton_raphson_scalar(f, fp, 1.5, 1e-6, 50, a, b)
        scipy_root = newton(lambda x: a*x**2 - b, 1.5, fprime=lambda x: 2*a*x)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 2.0, atol=1e-6)

    def test_three_parameters(self):
        """Test with three parameters (typical atmospheric equation)."""
        @njit
        def f(x, a, b, c):
            return a * x**2 + b * x + c
        
        @njit
        def fp(x, a, b, c):
            return 2 * a * x + b
        
        # Solve x^2 - 5x + 6 = 0, root at x=2 or x=3
        a, b, c = 1.0, -5.0, 6.0
        root, iters, converged = _newton_raphson_scalar(f, fp, 1.5, 1e-6, 50, a, b, c)
        scipy_root = newton(lambda x: a*x**2 + b*x + c, 1.5, 
                           fprime=lambda x: 2*a*x + b)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_four_parameters_atmospheric(self):
        """Test with four parameters (atmospheric application)."""
        @njit
        def f(T, T_surf, Td, P, factor):
            # Simplified atmospheric equation
            return T - Td - factor * (T_surf - T) * (P / 101325.0)
        
        @njit
        def fp(T, T_surf, Td, P, factor):
            return 1.0 + factor * (P / 101325.0)
        
        T_surf, Td, P, factor = 293.15, 283.15, 85000.0, 0.2
        root, iters, converged = _newton_raphson_scalar(
            f, fp, 285.0, 1e-6, 50, 
            T_surf, Td, P, factor
        )
        
        scipy_root = newton(
            lambda T: T - Td - factor * (T_surf - T) * (P / 101325.0),
            285.0,
            fprime=lambda T: 1.0 + factor * (P / 101325.0)
        )
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_five_parameters(self):
        """Test with five parameters."""
        @njit
        def f(x, p0, p1, p2, p3, p4):
            return p0*x**4 + p1*x**3 + p2*x**2 + p3*x + p4
        
        @njit
        def fp(x, p0, p1, p2, p3, p4):
            return 4*p0*x**3 + 3*p1*x**2 + 2*p2*x + p3
        
        params = (1.0, 0.0, -10.0, 0.0, 9.0)  # x^4 - 10x^2 + 9
        root, iters, converged = _newton_raphson_scalar(f, fp, 1.5, 1e-6, 50, *params)
        
        scipy_root = newton(
            lambda x: params[0]*x**4 + params[1]*x**3 + params[2]*x**2 + params[3]*x + params[4],
            1.5,
            fprime=lambda x: 4*params[0]*x**3 + 3*params[1]*x**2 + 2*params[2]*x + params[3]
        )
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_transcendental_with_params(self):
        """Test transcendental equation with parameters."""
        @njit
        def f(x, a, b):
            return a * np.sin(x) - b
        
        @njit
        def fp(x, a, b):
            return a * np.cos(x)
        
        a, b = 2.0, 1.0
        root, iters, converged = _newton_raphson_scalar(f, fp, 0.5, 1e-8, 50, a, b)
        scipy_root = newton(lambda x: a * np.sin(x) - b, 0.5, 
                           fprime=lambda x: a * np.cos(x))
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-8)

    def test_zero_derivative_handling(self):
        """Test handling of zero derivative (should not converge)."""
        @njit
        def f(x, k):
            return (x - k)**2
        
        @njit
        def fp(x, k):
            return 2 * (x - k)
        
        k = 5.0
        # Start at exactly k where derivative is zero
        root, iters, converged = _newton_raphson_scalar(f, fp, k, 1e-6, 50, k)
        
        assert not converged  # Should fail due to zero derivative

    def test_non_convergence(self):
        """Test equation that doesn't converge."""
        @njit
        def f(x, offset):
            return x**2 + offset  # No real roots for positive offset
        
        @njit
        def fp(x, offset):
            return 2 * x
        
        offset = 1.0
        root, iters, converged = _newton_raphson_scalar(f, fp, 1.0, 1e-6, 10, offset)
        
        # Should either not converge or have large function value
        assert (not converged) or (abs(f(root, offset)) > 1e-3)


class TestBisectionScalar:
    """Test bisection scalar solver with various parameter configurations."""

    def test_no_parameters(self):
        """Test with function requiring no parameters."""
        @njit
        def f(x):
            return x**2 - 4
        
        root, iters, converged = _bisection_scalar(f, 0.0, 5.0, 1e-6, 100)
        scipy_root = brentq(lambda x: x**2 - 4, 0.0, 5.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 2.0, atol=1e-6)

    def test_single_parameter(self):
        """Test with one parameter."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k = 27.0
        root, iters, converged = _bisection_scalar(f, 0.0, 5.0, 1e-6, 100, k)
        scipy_root = brentq(lambda x: x**3 - k, 0.0, 5.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 3.0, atol=1e-6)

    def test_two_parameters(self):
        """Test with two parameters."""
        @njit
        def f(x, a, b):
            return a * x**2 - b
        
        a, b = 1.0, 16.0
        root, iters, converged = _bisection_scalar(f, 0.0, 10.0, 1e-6, 100, a, b)
        scipy_root = brentq(lambda x: a * x**2 - b, 0.0, 10.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 4.0, atol=1e-6)

    def test_three_parameters(self):
        """Test with three parameters."""
        @njit
        def f(x, a, b, c):
            return a * x**3 + b * x + c
        
        a, b, c = 1.0, -2.0, -5.0
        root, iters, converged = _bisection_scalar(f, 2.0, 3.0, 1e-6, 100, a, b, c)
        scipy_root = brentq(lambda x: a*x**3 + b*x + c, 2.0, 3.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_atmospheric_equation_four_params(self):
        """Test with atmospheric-like equation (4 parameters)."""
        @njit
        def f(T, T_surf, Td, P, factor):
            return T - Td - factor * (T_surf - T) * (P / 101325.0)
        
        T_surf, Td, P, factor = 293.15, 283.15, 85000.0, 0.2
        root, iters, converged = _bisection_scalar(
            f, 250.0, 300.0, 1e-4, 100, 
            T_surf, Td, P, factor
        )
        
        scipy_root = brentq(
            lambda T: T - Td - factor * (T_surf - T) * (P / 101325.0),
            250.0, 300.0
        )
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-4)

    def test_invalid_bracket(self):
        """Test with invalid bracket (same sign at both ends)."""
        @njit
        def f(x, offset):
            return x**2 + offset
        
        offset = 1.0  # No roots for positive offset
        root, iters, converged = _bisection_scalar(f, 0.0, 5.0, 1e-6, 100, offset)
        
        assert not converged
        assert np.isnan(root)

    def test_root_at_boundary(self):
        """Test when root is exactly at bracket boundary."""
        @njit
        def f(x, k):
            return x - k
        
        k = 2.0
        root, iters, converged = _bisection_scalar(f, 2.0, 5.0, 1e-6, 100, k)
        
        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)


class TestBrentScalar:
    """Test Brent's method scalar solver with various parameter configurations."""

    def test_no_parameters(self):
        """Test with function requiring no parameters."""
        @njit
        def f(x):
            return x**2 - 4
        
        root, iters, converged = _brent_scalar(f, 0.0, 5.0, 1e-6, 100)
        scipy_root = brentq(lambda x: x**2 - 4, 0.0, 5.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_single_parameter(self):
        """Test with one parameter."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k = 8.0
        root, iters, converged = _brent_scalar(f, 0.0, 5.0, 1e-8, 100, k)
        scipy_root = brentq(lambda x: x**3 - k, 0.0, 5.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-8)
        assert np.isclose(root, 2.0, atol=1e-8)

    def test_two_parameters(self):
        """Test with two parameters."""
        @njit
        def f(x, a, b):
            return a * x**3 - b
        
        a, b = 1.0, 27.0
        root, iters, converged = _brent_scalar(f, 0.0, 5.0, 1e-8, 100, a, b)
        scipy_root = brentq(lambda x: a * x**3 - b, 0.0, 5.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-8)
        assert np.isclose(root, 3.0, atol=1e-8)

    def test_three_parameters(self):
        """Test with three parameters."""
        @njit
        def f(x, a, b, c):
            return a * x**3 + b * x + c
        
        a, b, c = 1.0, -2.0, -5.0
        root, iters, converged = _brent_scalar(f, 2.0, 3.0, 1e-8, 100, a, b, c)
        scipy_root = brentq(lambda x: a*x**3 + b*x + c, 2.0, 3.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-8)

    def test_transcendental_with_params(self):
        """Test transcendental equation with parameters."""
        @njit
        def f(x, scale, offset):
            return scale * np.sin(x) - offset
        
        scale, offset = 2.0, 1.0
        root, iters, converged = _brent_scalar(f, 0.0, np.pi/2, 1e-8, 100, scale, offset)
        scipy_root = brentq(lambda x: scale * np.sin(x) - offset, 0.0, np.pi/2)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-8)

    def test_efficiency_vs_bisection(self):
        """Verify Brent is more efficient than bisection with parameters."""
        @njit
        def f(x, k):
            return x**3 - k
        
        k = 10.0
        
        root_brent, iters_brent, conv_brent = _brent_scalar(
            f, 0.0, 5.0, 1e-8, 100, k
        )
        root_bisect, iters_bisect, conv_bisect = _bisection_scalar(
            f, 0.0, 5.0, 1e-8, 100, k
        )
        
        assert conv_brent and conv_bisect
        assert np.isclose(root_brent, root_bisect, atol=1e-8)
        assert iters_brent < iters_bisect  # Brent should converge faster


class TestParameterUnpackingEdgeCases:
    """Test edge cases for parameter unpacking in JIT functions."""

    def test_mixed_int_float_parameters(self):
        """Test mixing float and int parameters."""
        @njit
        def f(x, a, n):
            return a * x**n - 8.0
        
        @njit
        def fp(x, a, n):
            return a * n * x**(n-1)
        
        a, n = 1.0, 3  # Mixed float and int
        root, iters, converged = _newton_raphson_scalar(f, fp, 1.5, 1e-6, 50, a, n)
        
        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)

    def test_negative_parameters(self):
        """Test with negative parameter values."""
        @njit
        def f(x, offset):
            return x**2 + offset
        
        offset = -4.0  # Negative parameter
        root, iters, converged = _bisection_scalar(f, 0.0, 5.0, 1e-6, 100, offset)
        
        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)

    def test_zero_parameter(self):
        """Test with zero as parameter value."""
        @njit
        def f(x, offset):
            return x + offset
        
        offset = 0.0
        root, iters, converged = _bisection_scalar(f, -5.0, 5.0, 1e-6, 100, offset)
        
        assert converged
        assert np.isclose(root, 0.0, atol=1e-6)

    def test_large_parameter_values(self):
        """Test with large parameter values."""
        @njit
        def f(x, scale):
            return x**2 - scale
        
        scale = 1e6
        root, iters, converged = _bisection_scalar(f, 0.0, 2000.0, 1e-3, 100, scale)
        
        assert converged
        assert np.isclose(root, np.sqrt(scale), rtol=1e-5)

    def test_very_small_parameters(self):
        """Test with very small parameter values."""
        @njit
        def f(x, epsilon):
            return x - epsilon
        
        epsilon = 1e-10
        root, iters, converged = _brent_scalar(f, 0.0, 1e-5, 1e-12, 100, epsilon)
        
        assert converged
        assert np.isclose(root, epsilon, atol=1e-12)


class TestConsistencyWithScipy:
    """Verify all solvers produce results consistent with scipy across parameter ranges."""

    @pytest.mark.parametrize("k", [1.0, 5.0, 10.0, 100.0])
    def test_newton_consistency(self, k):
        """Test Newton-Raphson consistency across different parameter values."""
        @njit
        def f(x, param):
            return x**2 - param
        
        @njit
        def fp(x, param):
            return 2 * x
        
        x0 = np.sqrt(k) * 0.5
        root, _, converged = _newton_raphson_scalar(f, fp, x0, 1e-8, 50, k)
        scipy_root = newton(lambda x: x**2 - k, x0, fprime=lambda x: 2*x)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-8)

    @pytest.mark.parametrize("k", [5.0, 10.0, 27.0, 64.0])
    def test_brent_consistency(self, k):
        """Test Brent consistency across different parameter values."""
        @njit
        def f(x, param):
            return x**3 - param
        
        root, _, converged = _brent_scalar(f, 0.0, 10.0, 1e-8, 100, k)
        scipy_root = brentq(lambda x: x**3 - k, 0.0, 10.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-8)

    @pytest.mark.parametrize("offset", [-4.0, -9.0, -16.0, -25.0])
    def test_bisection_consistency(self, offset):
        """Test bisection consistency across different parameter values."""
        @njit
        def f(x, param):
            return x**2 + param
        
        root, _, converged = _bisection_scalar(
            f, 0.0, np.sqrt(-offset) + 1.0, 1e-6, 100, offset
        )
        scipy_root = brentq(lambda x: x**2 + offset, 0.0, np.sqrt(-offset) + 1.0)
        
        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

