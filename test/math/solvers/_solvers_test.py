"""
Unit tests for JIT solver implementations with SciPy validation.

Author: Cian Quezon
Tests accuracy against SciPy on challenging equations.
"""

import numpy as np
import pytest
from numba import njit
from scipy import optimize

# Import your solver classes
from meteorological_equations.math.solvers._solvers import (
    BisectionSolver,
    BrentSolver,
    NewtonRaphsonSolver,
)

# ============================================================================
# Test Functions (njit-compatible)
# ============================================================================


@njit
def polynomial_func(x: float) -> float:
    """Simple polynomial: x^3 - 2x - 5 = 0, root ≈ 2.0946"""
    return x**3 - 2 * x - 5


@njit
def polynomial_derivative(x: float) -> float:
    """Derivative of polynomial: 3x^2 - 2"""
    return 3 * x**2 - 2


@njit
def transcendental_func(x: float) -> float:
    """Challenging transcendental: x - cos(x) = 0, root ≈ 0.7391"""
    return x - np.cos(x)


@njit
def transcendental_derivative(x: float) -> float:
    """Derivative: 1 + sin(x)"""
    return 1 + np.sin(x)


@njit
def exponential_func(x: float) -> float:
    """Exponential equation: e^x - 3x = 0, root ≈ 1.512"""
    return np.exp(x) - 3 * x


@njit
def exponential_derivative(x: float) -> float:
    """Derivative: e^x - 3"""
    return np.exp(x) - 3


@njit
def steep_func(x: float) -> float:
    """Steep function: x^10 - 1 = 0, root = 1.0"""
    return x**10 - 1


@njit
def steep_derivative(x: float) -> float:
    """Derivative: 10x^9"""
    return 10 * x**9


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def newton_solver():
    """Fixture for NewtonRaphsonSolver instance"""
    return NewtonRaphsonSolver()


@pytest.fixture
def bisection_solver():
    """Fixture for BisectionSolver instance"""
    return BisectionSolver()


@pytest.fixture
def brent_solver():
    """Fixture for BrentSolver instance"""
    return BrentSolver()


# ============================================================================
# Accuracy Tests - Scalar
# ============================================================================


class TestScalarAccuracy:
    """Test scalar root finding accuracy against SciPy"""

    def test_newton_polynomial_scalar(self, newton_solver):
        """Newton-Raphson on polynomial equation"""
        root, iters, converged = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=2.0, tol=1e-6
        )
        scipy_root = optimize.newton(
            lambda x: x**3 - 2 * x - 5, x0=2.0, fprime=lambda x: 3 * x**2 - 2, tol=1e-6
        )

        assert converged, "Newton-Raphson failed to converge"
        assert np.abs(root - scipy_root) < 1e-6
        assert iters < 50

    def test_newton_transcendental_scalar(self, newton_solver):
        """Newton-Raphson on transcendental equation"""
        root, iters, converged = newton_solver.find_root(
            transcendental_func, transcendental_derivative, x0=0.5, tol=1e-6
        )
        scipy_root = optimize.newton(
            lambda x: x - np.cos(x), x0=0.5, fprime=lambda x: 1 + np.sin(x), tol=1e-6
        )

        assert converged
        assert np.abs(root - scipy_root) < 1e-6

    def test_bisection_polynomial_scalar(self, bisection_solver):
        """Bisection on polynomial equation"""
        root, iters, converged = bisection_solver.find_root(polynomial_func, a=2.0, b=3.0, tol=1e-6)
        scipy_root = optimize.bisect(lambda x: x**3 - 2 * x - 5, a=2.0, b=3.0, xtol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-5

    def test_brent_exponential_scalar(self, brent_solver):
        """Brent's method on exponential equation"""
        root, iters, converged = brent_solver.find_root(exponential_func, a=0.3, b=1.5, tol=1e-6)
        scipy_root = optimize.brentq(lambda x: np.exp(x) - 3 * x, a=0.3, b=1.5, xtol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-6


# ============================================================================
# Accuracy Tests - Vectorized
# ============================================================================


class TestVectorizedAccuracy:
    """Test vectorized root finding accuracy"""

    def test_newton_vectorized_polynomial(self, newton_solver):
        """Newton-Raphson vectorized on polynomial"""
        x0_array = np.array([1.5, 2.0, 2.5, 3.0])
        roots, iters, converged = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=x0_array, tol=1e-6
        )

        # Compare with SciPy individually
        scipy_roots = np.array(
            [
                optimize.newton(
                    lambda x: x**3 - 2 * x - 5, x0=x0, fprime=lambda x: 3 * x**2 - 2, tol=1e-6
                )
                for x0 in x0_array
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-6)

    def test_bisection_vectorized_transcendental(self, bisection_solver):
        """Bisection vectorized on transcendental"""
        a_array = np.array([0.0, 0.1, 0.2])
        b_array = np.array([1.0, 1.1, 1.2])

        roots, iters, converged = bisection_solver.find_root(
            transcendental_func, a=a_array, b=b_array, tol=1e-6
        )

        scipy_roots = np.array(
            [
                optimize.bisect(lambda x: x - np.cos(x), a=a, b=b, xtol=1e-6)
                for a, b in zip(a_array, b_array)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-5)

    def test_brent_vectorized_shape_preservation(self, brent_solver):
        """Brent's method preserves array shape"""
        a_array = np.array([[0.3, 0.3], [0.3, 0.3]])
        b_array = np.array([[1.5, 1.5], [1.5, 1.5]])

        roots, iters, converged = brent_solver.find_root(
            exponential_func, a=a_array, b=b_array, tol=1e-6
        )

        assert roots.shape == a_array.shape
        assert iters.shape == a_array.shape
        assert converged.shape == a_array.shape
        assert np.all(converged)


# ============================================================================
# Edge Cases and Robustness
# ============================================================================


class TestEdgeCases:
    """Test solver robustness on challenging cases"""

    def test_newton_steep_function(self, newton_solver):
        """Newton-Raphson on steep function"""
        root, iters, converged = newton_solver.find_root(
            steep_func, steep_derivative, x0=0.8, tol=1e-6, max_iter=100
        )

        assert converged
        assert np.abs(root - 1.0) < 1e-5

    def test_bisection_tight_bracket(self, bisection_solver):
        """Bisection with very tight initial bracket"""
        root, iters, converged = bisection_solver.find_root(
            polynomial_func, a=2.09, b=2.10, tol=1e-6
        )

        assert converged
        assert 2.09 <= root <= 2.10

    def test_convergence_flags(self, newton_solver):
        """Verify convergence flags are set correctly"""
        # Should converge
        root1, iters1, conv1 = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=2.0, max_iter=50
        )
        assert conv1 is True

        # Difficult case with low max_iter (may not converge)
        root2, iters2, conv2 = newton_solver.find_root(
            steep_func, steep_derivative, x0=0.1, max_iter=5
        )
        # Just verify the flag is returned (convergence depends on starting point)
        assert isinstance(conv2, (bool, np.bool_))


# ============================================================================
# Performance Tests (without pytest-benchmark)
# ============================================================================


class TestPerformance:
    """Performance tests on large arrays"""

    def test_newton_performance_large_array(self, newton_solver):
        """Test Newton-Raphson on large array (10K elements)"""
        x0_array = np.linspace(1.5, 3.0, 10000)

        roots, iters, converged = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=x0_array, tol=1e-6
        )

        assert np.mean(converged) > 0.95  # At least 95% convergence
        assert roots.shape == x0_array.shape
        assert iters.shape == x0_array.shape

    def test_bisection_performance_large_array(self, bisection_solver):
        """Test Bisection on large array (10K elements)"""
        a_array = np.full(10000, 2.0)
        b_array = np.full(10000, 3.0)

        roots, iters, converged = bisection_solver.find_root(
            polynomial_func, a=a_array, b=b_array, tol=1e-6
        )

        assert np.all(converged)
        assert roots.shape == a_array.shape

    def test_brent_performance_large_array(self, brent_solver):
        """Test Brent's method on large array (10K elements)"""
        a_array = np.full(10000, 0.3)
        b_array = np.full(10000, 1.5)

        roots, iters, converged = brent_solver.find_root(
            exponential_func, a=a_array, b=b_array, tol=1e-6
        )

        assert np.all(converged)
        assert roots.shape == a_array.shape


# ============================================================================
# Parametrized Tests for Multiple Equations
# ============================================================================


@pytest.mark.parametrize(
    "func,derivative,x0,expected_root",
    [
        (polynomial_func, polynomial_derivative, 2.0, 2.0946),
        (transcendental_func, transcendental_derivative, 0.5, 0.7391),
        (exponential_func, exponential_derivative, 1.5, 1.512),  # x0=1.5 converges to second root
    ],
)
def test_newton_multiple_equations(func, derivative, x0, expected_root):
    """Test Newton-Raphson on multiple equations"""
    solver = NewtonRaphsonSolver()
    root, iters, converged = solver.find_root(func, derivative, x0=x0, tol=1e-4)

    assert converged
    assert np.abs(root - expected_root) < 1e-2


@pytest.mark.parametrize(
    "func,a,b,expected_root",
    [
        (polynomial_func, 2.0, 3.0, 2.0946),
        (transcendental_func, 0.0, 1.0, 0.7391),
        (exponential_func, 0.3, 1.5, 0.6191),  # First root in this bracket
    ],
)
def test_brent_multiple_equations(func, a, b, expected_root):
    """Test Brent's method on multiple equations"""
    solver = BrentSolver()
    root, iters, converged = solver.find_root(func, a=a, b=b, tol=1e-4)

    assert converged
    assert np.abs(root - expected_root) < 1e-2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
