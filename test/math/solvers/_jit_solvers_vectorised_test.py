"""
Unit tests for root finding solvers.

Tests vectorized implementations against scipy.optimize as reference.

Author: Cian Quezon
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import bisect, brentq, newton
from numba import njit

from meteorological_equations.math.solvers._jit_solvers import (  # Replace with actual import
    _newton_raphson_vectorised,
    _bisection_vectorised,
    _brent_vectorised,
)


# ========================================================================
# TEST FUNCTIONS (ALL @njit)
# ========================================================================


@njit
def f_quadratic(x):
    """f(x) = x^2 - 2, root = sqrt(2)"""
    return x**2 - 2.0


@njit
def fp_quadratic(x):
    """Derivative of f_quadratic"""
    return 2.0 * x


@njit
def f_linear(x):
    """f(x) = 2x - 4, root = 2"""
    return 2.0 * x - 4.0


@njit
def fp_linear(x):
    """Derivative of f_linear"""
    return 2.0


@njit
def f_cubic(x):
    """f(x) = x^3"""
    return x**3


@njit
def fp_cubic(x):
    """Derivative of f_cubic"""
    return 3.0 * x**2


@njit
def f_no_root(x):
    """f(x) = x^2 + 1, no real roots"""
    return x**2 + 1.0


@njit
def f_simple_linear(x):
    """f(x) = x - 2, root = 2"""
    return x - 2.0


# ========================================================================
# NEWTON-RAPHSON TESTS
# ========================================================================


class TestNewtonRaphsonVectorised:
    """Tests for vectorized Newton-Raphson implementation."""

    def test_converges_to_sqrt2(self):
        """Test convergence to sqrt(2) with multiple initial guesses."""
        x0 = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
        expected = np.sqrt(2)

        roots, iters, converged = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0)

        assert np.all(converged), "All points should converge"
        assert_allclose(roots, expected, rtol=1e-6)

    def test_matches_scipy(self):
        """Compare results with scipy.optimize.newton."""
        x0 = np.array([1.0, 1.5, 2.0])

        # Our implementation
        roots_ours, _, converged = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0)

        # Scipy reference (NOT @njit)
        roots_scipy = np.array([newton(lambda x: x**2 - 2, x, fprime=lambda x: 2 * x) for x in x0])

        assert np.all(converged)
        assert_allclose(roots_ours, roots_scipy, rtol=1e-6)

    def test_handles_zero_derivative(self):
        """Test behavior when derivative is zero."""
        x0 = np.array([0.0, 0.1])  # Zero derivative at x=0

        roots, iters, converged = _newton_raphson_vectorised(f_cubic, fp_cubic, x0)

        # Should fail at x=0 (zero derivative), succeed at x=0.1
        assert not converged[0], "Should not converge at zero derivative"
        assert converged[1], "Should converge from x=0.1"

    def test_returns_correct_shapes(self):
        """Test that return arrays have correct shapes."""
        x0 = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

        roots, iters, converged = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0)

        assert roots.shape == x0.shape
        assert iters.shape == x0.shape
        assert converged.shape == x0.shape
        assert roots.dtype == np.float64
        assert iters.dtype == np.int64
        assert converged.dtype == bool  # Numba uses bool, not np.bool_

    def test_linear_function(self):
        """Test with simple linear function."""
        x0 = np.array([0.0, 1.0, 3.0, 5.0])
        expected = 2.0

        roots, iters, converged = _newton_raphson_vectorised(f_linear, fp_linear, x0)

        assert np.all(converged)
        assert_allclose(roots, expected, rtol=1e-6)


# ========================================================================
# BISECTION TESTS
# ========================================================================


class TestBisectionVectorised:
    """Tests for vectorized bisection implementation."""

    def test_converges_to_sqrt2(self):
        """Test convergence to sqrt(2) with multiple brackets."""
        n = 5
        a = np.zeros(n)
        b = np.full(n, 2.0)
        expected = np.sqrt(2)

        roots, iters, converged = _bisection_vectorised(f_quadratic, a, b)

        assert np.all(converged), "All points should converge"
        assert_allclose(roots, expected, rtol=1e-6)

    def test_matches_scipy(self):
        """Compare results with scipy.optimize.bisect."""
        a = np.array([0.0, 0.5, 1.0])
        b = np.array([2.0, 2.0, 2.0])

        # Our implementation
        roots_ours, _, converged = _bisection_vectorised(f_quadratic, a, b)

        # Scipy reference (NOT @njit)
        roots_scipy = np.array([bisect(lambda x: x**2 - 2, a[i], b[i]) for i in range(len(a))])

        assert np.all(converged)
        assert_allclose(roots_ours, roots_scipy, rtol=1e-6)

    def test_detects_no_bracketing(self):
        """Test detection when f(a) and f(b) have same sign."""
        a = np.array([0.0, 1.0])
        b = np.array([2.0, 3.0])

        roots, iters, converged = _bisection_vectorised(f_no_root, a, b)

        assert np.all(~converged), "Should not converge (no root)"
        assert np.all(np.isnan(roots)), "Should return NaN"

    def test_finds_root_at_boundary(self):
        """Test when root is exactly at bracket boundary."""
        a = np.array([0.0, 2.0])  # Root at b[0], root at a[1]
        b = np.array([2.0, 4.0])

        roots, iters, converged = _bisection_vectorised(f_simple_linear, a, b)

        assert np.all(converged)
        assert_allclose(roots, 2.0, rtol=1e-6)

    def test_different_brackets(self):
        """Test with various bracket widths."""
        a = np.array([0.0, 0.5, 1.0, 1.3])
        b = np.array([2.0, 2.0, 2.0, 2.0])
        expected = np.sqrt(2)

        roots, iters, converged = _bisection_vectorised(f_quadratic, a, b)

        assert np.all(converged)
        assert_allclose(roots, expected, rtol=1e-6)


# ========================================================================
# BRENT TESTS
# ========================================================================


class TestBrentVectorised:
    """Tests for vectorized Brent's method implementation."""

    def test_converges_to_sqrt2(self):
        """Test convergence to sqrt(2) with multiple brackets."""
        n = 5
        a = np.zeros(n)
        b = np.full(n, 2.0)
        expected = np.sqrt(2)

        roots, iters, converged = _brent_vectorised(f_quadratic, a, b)

        assert np.all(converged), "All points should converge"
        assert_allclose(roots, expected, rtol=1e-6)

    def test_matches_scipy(self):
        """Compare results with scipy.optimize.brentq."""
        a = np.array([0.0, 0.5, 1.0])
        b = np.array([2.0, 2.0, 2.0])

        # Our implementation
        roots_ours, _, converged = _brent_vectorised(f_quadratic, a, b)

        # Scipy reference (NOT @njit)
        roots_scipy = np.array([brentq(lambda x: x**2 - 2, a[i], b[i]) for i in range(len(a))])

        assert np.all(converged)
        assert_allclose(roots_ours, roots_scipy, rtol=1e-6)

    def test_faster_than_bisection(self):
        """Brent should converge in fewer iterations than bisection."""
        n = 10
        a = np.zeros(n)
        b = np.full(n, 2.0)

        _, iters_brent, _ = _brent_vectorised(f_quadratic, a, b)
        _, iters_bisect, _ = _bisection_vectorised(f_quadratic, a, b)

        # Brent should generally use fewer iterations
        assert np.mean(iters_brent) < np.mean(iters_bisect)

    def test_handles_no_bracketing(self):
        """Test when no root is bracketed."""
        a = np.array([0.0, 1.0])
        b = np.array([2.0, 3.0])

        roots, iters, converged = _brent_vectorised(f_no_root, a, b)

        assert np.all(~converged)
        assert np.all(np.isnan(roots))


# ========================================================================
# COMPARISON TESTS
# ========================================================================


class TestMethodComparison:
    """Compare all three methods against each other."""

    def test_all_find_same_root(self):
        """All methods should find the same root."""
        expected = np.sqrt(2)

        # Newton-Raphson
        x0_newton = np.array([1.0, 1.5, 2.0])
        roots_nr, _, conv_nr = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0_newton)

        # Bisection
        a = np.zeros(3)
        b = np.full(3, 2.0)
        roots_bisect, _, conv_bisect = _bisection_vectorised(f_quadratic, a, b)

        # Brent
        roots_brent, _, conv_brent = _brent_vectorised(f_quadratic, a, b)

        # All should converge
        assert np.all(conv_nr) and np.all(conv_bisect) and np.all(conv_brent)

        # All should find same root
        assert_allclose(roots_nr, expected, rtol=1e-6)
        assert_allclose(roots_bisect, expected, rtol=1e-6)
        assert_allclose(roots_brent, expected, rtol=1e-6)

    def test_iteration_counts(self):
        """Compare iteration counts: Newton < Brent < Bisection."""
        # Newton-Raphson
        x0 = np.full(10, 1.5)
        _, iters_nr, _ = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0)

        # Bisection and Brent
        a = np.zeros(10)
        b = np.full(10, 2.0)
        _, iters_bisect, _ = _bisection_vectorised(f_quadratic, a, b)
        _, iters_brent, _ = _brent_vectorised(f_quadratic, a, b)

        # Expected ordering
        assert np.mean(iters_nr) < np.mean(iters_brent)
        assert np.mean(iters_brent) < np.mean(iters_bisect)


# ========================================================================
# EDGE CASES
# ========================================================================


class TestEdgeCases:
    """Test edge cases and corner conditions."""

    def test_single_element_arrays(self):
        """Test with single-element arrays."""
        x0 = np.array([1.5])
        expected = np.sqrt(2)

        roots, iters, converged = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0)

        assert roots.shape == (1,)
        assert converged[0]
        assert_allclose(roots[0], expected, rtol=1e-6)

    def test_large_arrays(self):
        """Test with large arrays (performance check)."""
        n = 1000  # Reduced from 10000 for faster tests
        x0 = np.random.uniform(0.5, 3.0, n)
        expected = np.sqrt(2)

        roots, iters, converged = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0)

        # Most should converge
        convergence_rate = np.sum(converged) / n
        assert convergence_rate > 0.95, f"Only {convergence_rate * 100:.1f}% converged"
        assert_allclose(roots[converged], expected, rtol=1e-6)

    def test_tolerance_parameter(self):
        """Test that tighter tolerance requires more iterations."""
        x0 = np.array([1.5])

        # Loose tolerance
        roots_loose, iters_loose, conv_loose = _newton_raphson_vectorised(
            f_quadratic, fp_quadratic, x0, tol=1e-3
        )

        # Tight tolerance
        roots_tight, iters_tight, conv_tight = _newton_raphson_vectorised(
            f_quadratic, fp_quadratic, x0, tol=1e-9
        )

        # Both should converge
        assert conv_loose[0] and conv_tight[0]

        # Tight tolerance should require more (or equal) iterations
        assert iters_tight[0] >= iters_loose[0]

        # Both should be close to sqrt(2)
        expected = np.sqrt(2)
        assert_allclose(roots_loose[0], expected, rtol=1e-3)
        assert_allclose(roots_tight[0], expected, rtol=1e-9)

    def test_various_array_sizes(self):
        """Test with various array sizes."""
        for n in [1, 5, 10, 50, 100]:
            x0 = np.full(n, 1.5)
            roots, iters, converged = _newton_raphson_vectorised(f_quadratic, fp_quadratic, x0)

            assert roots.shape == (n,)
            assert iters.shape == (n,)
            assert converged.shape == (n,)
            assert np.all(converged)


# ========================================================================
# RUN TESTS
# ========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
