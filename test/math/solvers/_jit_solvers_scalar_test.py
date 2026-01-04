"""
Unit tests for root finding solvers with Numba JIT compilation.

Tests use scipy as reference implementation to validate correctness.
"""

import numpy as np
import pytest
from scipy import optimize
from numba import njit

from meteorological_equations.math.solvers._jit_solvers import (
    _newton_raphson_scalar,
    _bisection_scalar,
    _brent_scalar,
)


# Test functions and their derivatives - must be JIT-compiled for Numba
@njit
def linear(x):
    """f(x) = 2x - 4, root at x=2"""
    return 2 * x - 4


@njit
def linear_prime(x):
    return 2.0


@njit
def quadratic(x):
    """f(x) = x^2 - 4, roots at x=±2"""
    return x**2 - 4


@njit
def quadratic_prime(x):
    return 2 * x


@njit
def cubic(x):
    """f(x) = x^3 - x, roots at x=-1, 0, 1"""
    return x**3 - x


@njit
def cubic_prime(x):
    return 3 * x**2 - 1


@njit
def transcendental(x):
    """f(x) = x - cos(x), root near x≈0.739"""
    return x - np.cos(x)


@njit
def transcendental_prime(x):
    return 1 + np.sin(x)


@njit
def exponential(x):
    """f(x) = e^x - 2, root at x=ln(2)≈0.693"""
    return np.exp(x) - 2


@njit
def exponential_prime(x):
    return np.exp(x)


# Non-JIT versions for scipy comparison
def linear_nojit(x):
    return 2 * x - 4


def linear_prime_nojit(x):
    return 2.0


def quadratic_nojit(x):
    return x**2 - 4


def quadratic_prime_nojit(x):
    return 2 * x


def cubic_nojit(x):
    return x**3 - x


def transcendental_nojit(x):
    return x - np.cos(x)


def transcendental_prime_nojit(x):
    return 1 + np.sin(x)


def exponential_nojit(x):
    return np.exp(x) - 2


def exponential_prime_nojit(x):
    return np.exp(x)


class TestNewtonRaphson:
    """Test suite for Newton-Raphson method."""

    def test_linear_function(self):
        """Test with simple linear function."""
        root, iters, converged = _newton_raphson_scalar(linear, linear_prime, x0=1.0)
        scipy_root = optimize.newton(linear_nojit, x0=1.0, fprime=linear_prime_nojit)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert iters < 10

    def test_quadratic_positive_root(self):
        """Test quadratic function with positive initial guess."""
        root, iters, converged = _newton_raphson_scalar(quadratic, quadratic_prime, x0=3.0)
        scipy_root = optimize.newton(quadratic_nojit, x0=3.0, fprime=quadratic_prime_nojit)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_negative_root(self):
        """Test quadratic function with negative initial guess."""
        root, iters, converged = _newton_raphson_scalar(quadratic, quadratic_prime, x0=-3.0)
        scipy_root = optimize.newton(quadratic_nojit, x0=-3.0, fprime=quadratic_prime_nojit)

        assert converged
        assert np.isclose(root, -2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_transcendental_function(self):
        """Test with transcendental function x - cos(x)."""
        root, iters, converged = _newton_raphson_scalar(
            transcendental, transcendental_prime, x0=1.0
        )
        scipy_root = optimize.newton(
            transcendental_nojit, x0=1.0, fprime=transcendental_prime_nojit
        )

        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 0.7390851332, atol=1e-6)

    def test_exponential_function(self):
        """Test with exponential function."""
        root, iters, converged = _newton_raphson_scalar(exponential, exponential_prime, x0=1.0)
        scipy_root = optimize.newton(exponential_nojit, x0=1.0, fprime=exponential_prime_nojit)

        assert converged
        assert np.isclose(root, np.log(2), atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    """
Unit tests for root finding solvers with Numba JIT compilation.

Tests use scipy as reference implementation to validate correctness.
"""


import numpy as np
import pytest
from scipy import optimize
from numba import njit

from meteorological_equations.math.solvers._jit_solvers import (
    _newton_raphson_scalar,
    _bisection_scalar,
    _brent_scalar,
)


# Test functions and their derivatives - must be JIT-compiled for Numba
@njit
def linear(x):
    """f(x) = 2x - 4, root at x=2"""
    return 2 * x - 4


@njit
def linear_prime(x):
    return 2.0


@njit
def quadratic(x):
    """f(x) = x^2 - 4, roots at x=±2"""
    return x**2 - 4


@njit
def quadratic_prime(x):
    return 2 * x


@njit
def cubic(x):
    """f(x) = x^3 - x, roots at x=-1, 0, 1"""
    return x**3 - x


@njit
def cubic_prime(x):
    return 3 * x**2 - 1


@njit
def transcendental(x):
    """f(x) = x - cos(x), root near x≈0.739"""
    return x - np.cos(x)


@njit
def transcendental_prime(x):
    return 1 + np.sin(x)


@njit
def exponential(x):
    """f(x) = e^x - 2, root at x=ln(2)≈0.693"""
    return np.exp(x) - 2


@njit
def exponential_prime(x):
    return np.exp(x)


# Non-JIT versions for scipy comparison
def linear_nojit(x):
    return 2 * x - 4


def linear_prime_nojit(x):
    return 2.0


def quadratic_nojit(x):
    return x**2 - 4


def quadratic_prime_nojit(x):
    return 2 * x


def cubic_nojit(x):
    return x**3 - x


def transcendental_nojit(x):
    return x - np.cos(x)


def transcendental_prime_nojit(x):
    return 1 + np.sin(x)


def exponential_nojit(x):
    return np.exp(x) - 2


def exponential_prime_nojit(x):
    return np.exp(x)


class TestNewtonRaphson:
    """Test suite for Newton-Raphson method."""

    def test_linear_function(self):
        """Test with simple linear function."""
        root, iters, converged = _newton_raphson_scalar(linear, linear_prime, x0=1.0)
        scipy_root = optimize.newton(linear_nojit, x0=1.0, fprime=linear_prime_nojit)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert iters < 10

    def test_quadratic_positive_root(self):
        """Test quadratic function with positive initial guess."""
        root, iters, converged = _newton_raphson_scalar(quadratic, quadratic_prime, x0=3.0)
        scipy_root = optimize.newton(quadratic_nojit, x0=3.0, fprime=quadratic_prime_nojit)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_negative_root(self):
        """Test quadratic function with negative initial guess."""
        root, iters, converged = _newton_raphson_scalar(quadratic, quadratic_prime, x0=-3.0)
        scipy_root = optimize.newton(quadratic_nojit, x0=-3.0, fprime=quadratic_prime_nojit)

        assert converged
        assert np.isclose(root, -2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_transcendental_function(self):
        """Test with transcendental function x - cos(x)."""
        root, iters, converged = _newton_raphson_scalar(
            transcendental, transcendental_prime, x0=1.0
        )
        scipy_root = optimize.newton(
            transcendental_nojit, x0=1.0, fprime=transcendental_prime_nojit
        )

        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 0.7390851332, atol=1e-6)

    def test_exponential_function(self):
        """Test with exponential function."""
        root, iters, converged = _newton_raphson_scalar(exponential, exponential_prime, x0=1.0)
        scipy_root = optimize.newton(exponential_nojit, x0=1.0, fprime=exponential_prime_nojit)

        assert converged
        assert np.isclose(root, np.log(2), atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_zero_derivative(self):
        """Test behavior when derivative is zero."""

        @njit
        def flat_func(x):
            return (x - 1) ** 2

        @njit
        def flat_prime(x):
            return 2 * (x - 1)

        # Starting at x=1 where derivative is zero
        root, iters, converged = _newton_raphson_scalar(flat_func, flat_prime, x0=1.0)

        assert not converged  # Should fail to converge

    def test_max_iterations(self):
        """Test that max iterations limit is respected."""
        root, iters, converged = _newton_raphson_scalar(
            quadratic, quadratic_prime, x0=100.0, max_iter=5
        )

        assert iters <= 5

    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        tol = 1e-10
        root, iters, converged = _newton_raphson_scalar(quadratic, quadratic_prime, x0=3.0, tol=tol)

        assert converged
        assert abs(quadratic(root)) < tol * 10  # Function value should be very small


class TestBisection:
    """Test suite for bisection method."""

    def test_linear_function(self):
        """Test with simple linear function."""
        root, iters, converged = _bisection_scalar(linear, a=0.0, b=5.0)
        scipy_root = optimize.bisect(linear_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_positive_root(self):
        """Test quadratic with bracket around positive root."""
        root, iters, converged = _bisection_scalar(quadratic, a=0.0, b=5.0)
        scipy_root = optimize.bisect(quadratic_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_negative_root(self):
        """Test quadratic with bracket around negative root."""
        root, iters, converged = _bisection_scalar(quadratic, a=-5.0, b=0.0)
        scipy_root = optimize.bisect(quadratic_nojit, a=-5.0, b=0.0)

        assert converged
        assert np.isclose(root, -2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_transcendental_function(self):
        """Test with transcendental function."""
        root, iters, converged = _bisection_scalar(transcendental, a=0.0, b=1.0)
        scipy_root = optimize.bisect(transcendental_nojit, a=0.0, b=1.0)

        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_cubic_middle_root(self):
        """Test cubic function for middle root."""
        root, iters, converged = _bisection_scalar(cubic, a=-0.5, b=0.5)
        scipy_root = optimize.bisect(cubic_nojit, a=-0.5, b=0.5)

        assert converged
        assert np.isclose(root, 0.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_root_at_boundary(self):
        """Test when root is exactly at bracket boundary."""
        root, iters, converged = _bisection_scalar(quadratic, a=2.0, b=5.0)

        assert converged
        assert iters == 0
        assert root == 2.0

    def test_no_sign_change(self):
        """Test behavior when bracket doesn't contain root."""
        root, iters, converged = _bisection_scalar(quadratic, a=3.0, b=5.0)

        assert not converged
        assert np.isnan(root)

    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        tol = 1e-10
        root, iters, converged = _bisection_scalar(quadratic, a=0.0, b=5.0, tol=tol)

        assert converged
        assert abs(quadratic(root)) < tol * 10


class TestBrent:
    """Test suite for Brent's method."""

    def test_linear_function(self):
        """Test with simple linear function."""
        root, iters, converged = _brent_scalar(linear, a=0.0, b=5.0)
        scipy_root = optimize.brentq(linear_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_positive_root(self):
        """Test quadratic with bracket around positive root."""
        root, iters, converged = _brent_scalar(quadratic, a=0.0, b=5.0)
        scipy_root = optimize.brentq(quadratic_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_negative_root(self):
        """Test quadratic with bracket around negative root."""
        root, iters, converged = _brent_scalar(quadratic, a=-5.0, b=0.0)
        scipy_root = optimize.brentq(quadratic_nojit, a=-5.0, b=0.0)

        assert converged
        assert np.isclose(root, -2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_transcendental_function(self):
        """Test with transcendental function."""
        root, iters, converged = _brent_scalar(transcendental, a=0.0, b=1.0)
        scipy_root = optimize.brentq(transcendental_nojit, a=0.0, b=1.0)

        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 0.7390851332, atol=1e-6)

    def test_exponential_function(self):
        """Test with exponential function."""
        root, iters, converged = _brent_scalar(exponential, a=0.0, b=2.0)
        scipy_root = optimize.brentq(exponential_nojit, a=0.0, b=2.0)

        assert converged
        assert np.isclose(root, np.log(2), atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_cubic_function(self):
        """Test cubic function for middle root."""
        root, iters, converged = _brent_scalar(cubic, a=-0.5, b=0.5)
        scipy_root = optimize.brentq(cubic_nojit, a=-0.5, b=0.5)

        assert converged
        assert np.isclose(root, 0.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_root_at_boundary(self):
        """Test when root is exactly at bracket boundary."""
        root, iters, converged = _brent_scalar(quadratic, a=2.0, b=5.0)

        assert converged
        assert iters == 0
        assert root == 2.0

    def test_no_sign_change(self):
        """Test behavior when bracket doesn't contain root."""
        root, iters, converged = _brent_scalar(quadratic, a=3.0, b=5.0)

        assert not converged
        assert np.isnan(root)

    def test_convergence_speed(self):
        """Test that Brent's method converges faster than bisection."""
        # Brent should be faster than bisection for smooth functions
        root_brent, iters_brent, _ = _brent_scalar(exponential, a=0.0, b=2.0)
        root_bisect, iters_bisect, _ = _bisection_scalar(exponential, a=0.0, b=2.0)

        assert np.isclose(root_brent, root_bisect, atol=1e-6)
        assert iters_brent <= iters_bisect

    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        tol = 1e-10
        root, iters, converged = _brent_scalar(quadratic, a=0.0, b=5.0, tol=tol)

        assert converged
        assert abs(quadratic(root)) < tol * 10


class TestComparison:
    """Comparative tests across all methods."""

    def test_all_methods_agree_on_quadratic(self):
        """Test that all methods find the same root for quadratic."""
        root_nr, _, _ = _newton_raphson_scalar(quadratic, quadratic_prime, x0=3.0)
        root_bisect, _, _ = _bisection_scalar(quadratic, a=0.0, b=5.0)
        root_brent, _, _ = _brent_scalar(quadratic, a=0.0, b=5.0)

        assert np.isclose(root_nr, root_bisect, atol=1e-5)
        assert np.isclose(root_bisect, root_brent, atol=1e-5)
        assert np.isclose(root_nr, 2.0, atol=1e-6)

    def test_all_methods_agree_on_transcendental(self):
        """Test that all methods find the same root for transcendental function."""
        root_nr, _, _ = _newton_raphson_scalar(transcendental, transcendental_prime, x0=1.0)
        root_bisect, _, _ = _bisection_scalar(transcendental, a=0.0, b=1.0)
        root_brent, _, _ = _brent_scalar(transcendental, a=0.0, b=1.0)

        assert np.isclose(root_nr, root_bisect, atol=1e-5)
        assert np.isclose(root_bisect, root_brent, atol=1e-5)
        assert np.isclose(root_nr, 0.7390851332, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

    def test_zero_derivative(self):
        """Test behavior when derivative is zero."""

        @njit
        def flat_func(x):
            return (x - 1) ** 2

        @njit
        def flat_prime(x):
            return 2 * (x - 1)

        # Starting at x=1 where derivative is zero
        root, iters, converged = _newton_raphson_scalar(flat_func, flat_prime, x0=1.0)

        assert not converged  # Should fail to converge

    def test_max_iterations(self):
        """Test that max iterations limit is respected."""
        root, iters, converged = _newton_raphson_scalar(
            quadratic, quadratic_prime, x0=100.0, max_iter=5
        )

        assert iters <= 5

    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        tol = 1e-10
        root, iters, converged = _newton_raphson_scalar(quadratic, quadratic_prime, x0=3.0, tol=tol)

        assert converged
        assert abs(quadratic(root)) < tol * 10  # Function value should be very small


class TestBisection:
    """Test suite for bisection method."""

    def test_linear_function(self):
        """Test with simple linear function."""
        root, iters, converged = _bisection_scalar(linear, a=0.0, b=5.0)
        scipy_root = optimize.bisect(linear_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_positive_root(self):
        """Test quadratic with bracket around positive root."""
        root, iters, converged = _bisection_scalar(quadratic, a=0.0, b=5.0)
        scipy_root = optimize.bisect(quadratic_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_negative_root(self):
        """Test quadratic with bracket around negative root."""
        root, iters, converged = _bisection_scalar(quadratic, a=-5.0, b=0.0)
        scipy_root = optimize.bisect(quadratic_nojit, a=-5.0, b=0.0)

        assert converged
        assert np.isclose(root, -2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_transcendental_function(self):
        """Test with transcendental function."""
        root, iters, converged = _bisection_scalar(transcendental, a=0.0, b=1.0)
        scipy_root = optimize.bisect(transcendental_nojit, a=0.0, b=1.0)

        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_cubic_middle_root(self):
        """Test cubic function for middle root."""
        root, iters, converged = _bisection_scalar(cubic, a=-0.5, b=0.5)
        scipy_root = optimize.bisect(cubic_nojit, a=-0.5, b=0.5)

        assert converged
        assert np.isclose(root, 0.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_root_at_boundary(self):
        """Test when root is exactly at bracket boundary."""
        root, iters, converged = _bisection_scalar(quadratic, a=2.0, b=5.0)

        assert converged
        assert iters == 0
        assert root == 2.0

    def test_no_sign_change(self):
        """Test behavior when bracket doesn't contain root."""
        root, iters, converged = _bisection_scalar(quadratic, a=3.0, b=5.0)

        assert not converged
        assert np.isnan(root)

    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        tol = 1e-10
        root, iters, converged = _bisection_scalar(quadratic, a=0.0, b=5.0, tol=tol)

        assert converged
        assert abs(quadratic(root)) < tol * 10


class TestBrent:
    """Test suite for Brent's method."""

    def test_linear_function(self):
        """Test with simple linear function."""
        root, iters, converged = _brent_scalar(linear, a=0.0, b=5.0)
        scipy_root = optimize.brentq(linear_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_positive_root(self):
        """Test quadratic with bracket around positive root."""
        root, iters, converged = _brent_scalar(quadratic, a=0.0, b=5.0)
        scipy_root = optimize.brentq(quadratic_nojit, a=0.0, b=5.0)

        assert converged
        assert np.isclose(root, 2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_quadratic_negative_root(self):
        """Test quadratic with bracket around negative root."""
        root, iters, converged = _brent_scalar(quadratic, a=-5.0, b=0.0)
        scipy_root = optimize.brentq(quadratic_nojit, a=-5.0, b=0.0)

        assert converged
        assert np.isclose(root, -2.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_transcendental_function(self):
        """Test with transcendental function."""
        root, iters, converged = _brent_scalar(transcendental, a=0.0, b=1.0)
        scipy_root = optimize.brentq(transcendental_nojit, a=0.0, b=1.0)

        assert converged
        assert np.isclose(root, scipy_root, atol=1e-6)
        assert np.isclose(root, 0.7390851332, atol=1e-6)

    def test_exponential_function(self):
        """Test with exponential function."""
        root, iters, converged = _brent_scalar(exponential, a=0.0, b=2.0)
        scipy_root = optimize.brentq(exponential_nojit, a=0.0, b=2.0)

        assert converged
        assert np.isclose(root, np.log(2), atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_cubic_function(self):
        """Test cubic function for middle root."""
        root, iters, converged = _brent_scalar(cubic, a=-0.5, b=0.5)
        scipy_root = optimize.brentq(cubic_nojit, a=-0.5, b=0.5)

        assert converged
        assert np.isclose(root, 0.0, atol=1e-6)
        assert np.isclose(root, scipy_root, atol=1e-6)

    def test_root_at_boundary(self):
        """Test when root is exactly at bracket boundary."""
        root, iters, converged = _brent_scalar(quadratic, a=2.0, b=5.0)

        assert converged
        assert iters == 0
        assert root == 2.0

    def test_no_sign_change(self):
        """Test behavior when bracket doesn't contain root."""
        root, iters, converged = _brent_scalar(quadratic, a=3.0, b=5.0)

        assert not converged
        assert np.isnan(root)

    def test_convergence_speed(self):
        """Test that Brent's method converges faster than bisection."""
        # Brent should be faster than bisection for smooth functions
        root_brent, iters_brent, _ = _brent_scalar(exponential, a=0.0, b=2.0)
        root_bisect, iters_bisect, _ = _bisection_scalar(exponential, a=0.0, b=2.0)

        assert np.isclose(root_brent, root_bisect, atol=1e-6)
        assert iters_brent <= iters_bisect

    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        tol = 1e-10
        root, iters, converged = _brent_scalar(quadratic, a=0.0, b=5.0, tol=tol)

        assert converged
        assert abs(quadratic(root)) < tol * 10


class TestComparison:
    """Comparative tests across all methods."""

    def test_all_methods_agree_on_quadratic(self):
        """Test that all methods find the same root for quadratic."""
        root_nr, _, _ = _newton_raphson_scalar(quadratic, quadratic_prime, x0=3.0)
        root_bisect, _, _ = _bisection_scalar(quadratic, a=0.0, b=5.0)
        root_brent, _, _ = _brent_scalar(quadratic, a=0.0, b=5.0)

        assert np.isclose(root_nr, root_bisect, atol=1e-5)
        assert np.isclose(root_bisect, root_brent, atol=1e-5)
        assert np.isclose(root_nr, 2.0, atol=1e-6)

    def test_all_methods_agree_on_transcendental(self):
        """Test that all methods find the same root for transcendental function."""
        root_nr, _, _ = _newton_raphson_scalar(transcendental, transcendental_prime, x0=1.0)
        root_bisect, _, _ = _bisection_scalar(transcendental, a=0.0, b=1.0)
        root_brent, _, _ = _brent_scalar(transcendental, a=0.0, b=1.0)

        assert np.isclose(root_nr, root_bisect, atol=1e-5)
        assert np.isclose(root_bisect, root_brent, atol=1e-5)
        assert np.isclose(root_nr, 0.7390851332, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
