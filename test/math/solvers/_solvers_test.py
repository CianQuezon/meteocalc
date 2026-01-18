"""
Comprehensive integration tests for JIT solver implementations with SciPy validation.

Tests solver classes with scalar/vectorized inputs, parameter handling,
and validates accuracy against SciPy reference implementations.

Author: Cian Quezon
"""

import numpy as np
import pytest
from numba import njit
from scipy import optimize

from meteorological_equations.math.solvers._solvers import (
    BisectionSolver,
    BrentSolver,
    NewtonRaphsonSolver,
)

# ============================================================================
# Test Functions - No Parameters
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
# Test Functions - With Parameters
# ============================================================================


@njit
def func_one_param(x: float, k: float) -> float:
    """Function with one parameter: x^2 - k = 0"""
    return x**2 - k


@njit
def func_one_param_derivative(x: float, _k: float) -> float:
    """Derivative: 2x"""
    return 2 * x


@njit
def func_two_params(x: float, a: float, b: float) -> float:
    """Function with two parameters: a*x^2 - b = 0"""
    return a * x**2 - b


@njit
def func_two_params_derivative(x: float, a: float, _b: float) -> float:
    """Derivative: 2*a*x"""
    return 2 * a * x


@njit
def func_three_params(x: float, a: float, b: float, c: float) -> float:
    """Function with three parameters: a*x^2 + b*x + c = 0"""
    return a * x**2 + b * x + c


@njit
def func_three_params_derivative(x: float, a: float, b: float, _c: float) -> float:
    """Derivative: 2*a*x + b"""
    return 2 * a * x + b


@njit
def func_four_params(x: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """Function with four parameters: p0*x^3 + p1*x^2 + p2*x + p3 = 0"""
    return p0 * x**3 + p1 * x**2 + p2 * x + p3


@njit
def func_four_params_derivative(x: float, p0: float, p1: float, p2: float, _p3: float) -> float:
    """Derivative: 3*p0*x^2 + 2*p1*x + p2"""
    return 3 * p0 * x**2 + 2 * p1 * x + p2


@njit
def func_five_params(x: float, p0: float, p1: float, p2: float, p3: float, p4: float) -> float:
    """Function with five parameters: p0*x^4 + p1*x^3 + p2*x^2 + p3*x + p4 = 0"""
    return p0 * x**4 + p1 * x**3 + p2 * x**2 + p3 * x + p4


@njit
def func_five_params_derivative(
    x: float, p0: float, p1: float, p2: float, p3: float, _p4: float
) -> float:
    """Derivative: 4*p0*x^3 + 3*p1*x^2 + 2*p2*x + p3"""
    return 4 * p0 * x**3 + 3 * p1 * x**2 + 2 * p2 * x + p3


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
# Scalar Tests - No Parameters
# ============================================================================


class TestScalarNoParameters:
    """Test scalar solving without function parameters"""

    def test_newton_polynomial_scalar(self, newton_solver):
        """Newton-Raphson on polynomial equation (no params)"""
        root, iters, converged = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=2.0, tol=1e-6
        )
        scipy_root = optimize.newton(
            lambda x: x**3 - 2 * x - 5, x0=2.0, fprime=lambda x: 3 * x**2 - 2, tol=1e-6
        )

        assert converged
        assert np.abs(root - scipy_root) < 1e-6
        assert iters < 50

    def test_newton_transcendental_scalar(self, newton_solver):
        """Newton-Raphson on transcendental equation (no params)"""
        root, iters, converged = newton_solver.find_root(
            transcendental_func, transcendental_derivative, x0=0.5, tol=1e-6
        )
        scipy_root = optimize.newton(
            lambda x: x - np.cos(x), x0=0.5, fprime=lambda x: 1 + np.sin(x), tol=1e-6
        )

        assert converged
        assert np.abs(root - scipy_root) < 1e-6

    def test_bisection_polynomial_scalar(self, bisection_solver):
        """Bisection on polynomial equation (no params)"""
        root, iters, converged = bisection_solver.find_root(polynomial_func, a=2.0, b=3.0, tol=1e-6)
        scipy_root = optimize.bisect(lambda x: x**3 - 2 * x - 5, a=2.0, b=3.0, xtol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-5

    def test_brent_exponential_scalar(self, brent_solver):
        """Brent's method on exponential equation (no params)"""
        root, iters, converged = brent_solver.find_root(exponential_func, a=0.3, b=1.5, tol=1e-6)
        scipy_root = optimize.brentq(lambda x: np.exp(x) - 3 * x, a=0.3, b=1.5, xtol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-6


# ============================================================================
# Scalar Tests - With Parameters
# ============================================================================


class TestScalarWithParameters:
    """Test scalar solving with function parameters (1-5 params)"""

    def test_newton_one_parameter(self, newton_solver):
        """Newton-Raphson with 1 parameter"""
        k = 9.0
        root, iters, converged = newton_solver.find_root(
            func_one_param, func_one_param_derivative, x0=2.0, func_params=(k,), tol=1e-6
        )
        scipy_root = optimize.newton(lambda x: x**2 - k, x0=2.0, fprime=lambda x: 2 * x, tol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-6
        assert np.isclose(root, 3.0, atol=1e-6)

    def test_newton_two_parameters(self, newton_solver):
        """Newton-Raphson with 2 parameters"""
        a, b = 2.0, 8.0
        root, iters, converged = newton_solver.find_root(
            func_two_params, func_two_params_derivative, x0=1.5, func_params=(a, b), tol=1e-6
        )
        scipy_root = optimize.newton(
            lambda x: a * x**2 - b, x0=1.5, fprime=lambda x: 2 * a * x, tol=1e-6
        )

        assert converged
        assert np.abs(root - scipy_root) < 1e-6
        assert np.isclose(root, 2.0, atol=1e-6)

    def test_newton_three_parameters(self, newton_solver):
        """Newton-Raphson with 3 parameters"""
        a, b, c = 1.0, -5.0, 6.0  # x^2 - 5x + 6 = 0, roots at 2 and 3
        root, iters, converged = newton_solver.find_root(
            func_three_params, func_three_params_derivative, x0=1.5, func_params=(a, b, c), tol=1e-6
        )
        scipy_root = optimize.newton(
            lambda x: a * x**2 + b * x + c, x0=1.5, fprime=lambda x: 2 * a * x + b, tol=1e-6
        )

        assert converged
        assert np.abs(root - scipy_root) < 1e-6

    def test_newton_four_parameters(self, newton_solver):
        """Newton-Raphson with 4 parameters"""
        p0, p1, p2, p3 = 1.0, 0.0, -10.0, 9.0  # x^3 - 10x + 9 = 0
        root, iters, converged = newton_solver.find_root(
            func_four_params,
            func_four_params_derivative,
            x0=0.5,
            func_params=(p0, p1, p2, p3),
            tol=1e-6,
        )
        scipy_root = optimize.newton(
            lambda x: p0 * x**3 + p1 * x**2 + p2 * x + p3,
            x0=0.5,
            fprime=lambda x: 3 * p0 * x**2 + 2 * p1 * x + p2,
            tol=1e-6,
        )

        assert converged
        assert np.abs(root - scipy_root) < 1e-6

    def test_newton_five_parameters(self, newton_solver):
        """Newton-Raphson with 5 parameters"""
        p0, p1, p2, p3, p4 = 1.0, 0.0, -10.0, 0.0, 9.0  # x^4 - 10x^2 + 9 = 0
        root, iters, converged = newton_solver.find_root(
            func_five_params,
            func_five_params_derivative,
            x0=1.5,
            func_params=(p0, p1, p2, p3, p4),
            tol=1e-6,
        )
        scipy_root = optimize.newton(
            lambda x: p0 * x**4 + p1 * x**3 + p2 * x**2 + p3 * x + p4,
            x0=1.5,
            fprime=lambda x: 4 * p0 * x**3 + 3 * p1 * x**2 + 2 * p2 * x + p3,
            tol=1e-6,
        )

        assert converged
        assert np.abs(root - scipy_root) < 1e-6

    def test_bisection_one_parameter(self, bisection_solver):
        """Bisection with 1 parameter"""
        k = 27.0
        root, iters, converged = bisection_solver.find_root(
            func_one_param, a=0.0, b=10.0, func_params=(k,), tol=1e-6
        )
        scipy_root = optimize.bisect(lambda x: x**2 - k, a=0.0, b=10.0, xtol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-5
        assert np.isclose(root, np.sqrt(27.0), atol=1e-5)

    def test_bisection_three_parameters(self, bisection_solver):
        """Bisection with 3 parameters"""
        a, b, c = 1.0, -5.0, 6.0
        root, iters, converged = bisection_solver.find_root(
            func_three_params, a=1.5, b=2.5, func_params=(a, b, c), tol=1e-6
        )
        scipy_root = optimize.bisect(lambda x: a * x**2 + b * x + c, a=1.5, b=2.5, xtol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-5

    def test_brent_two_parameters(self, brent_solver):
        """Brent with 2 parameters"""
        a, b = 1.0, 16.0
        root, iters, converged = brent_solver.find_root(
            func_two_params, a=0.0, b=10.0, func_params=(a, b), tol=1e-6
        )
        scipy_root = optimize.brentq(lambda x: a * x**2 - b, a=0.0, b=10.0, xtol=1e-6)

        assert converged
        assert np.abs(root - scipy_root) < 1e-6
        assert np.isclose(root, 4.0, atol=1e-6)


# ============================================================================
# Vectorized Tests - No Parameters
# ============================================================================


class TestVectorizedNoParameters:
    """Test vectorized solving without function parameters"""

    def test_newton_vectorized_polynomial(self, newton_solver):
        """Newton-Raphson vectorized on polynomial (no params)"""
        x0_array = np.array([1.5, 2.0, 2.5, 3.0])
        roots, iters, converged = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=x0_array, tol=1e-6
        )

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
        """Bisection vectorized on transcendental (no params)"""
        a_array = np.array([0.0, 0.1, 0.2, 0.3])
        b_array = np.array([1.0, 1.1, 1.2, 1.3])

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
        """Brent's method preserves array shape (no params)"""
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
# Vectorized Tests - With Parameters
# ============================================================================


class TestVectorizedWithParameters:
    """Test vectorized solving with function parameters"""

    def test_newton_vectorized_one_parameter(self, newton_solver):
        """Newton-Raphson vectorized with 1 parameter per solve"""
        k_values = np.array([4.0, 9.0, 16.0, 25.0, 36.0])
        x0_array = np.sqrt(k_values) * 0.8

        roots, iters, converged = newton_solver.find_root(
            func_one_param,
            func_one_param_derivative,
            x0=x0_array,
            func_params=k_values,
            tol=1e-6,
        )

        scipy_roots = np.array(
            [
                optimize.newton(
                    lambda x, k=k: x**2 - k, x0=x0_array[i], fprime=lambda x: 2 * x, tol=1e-6
                )
                for i, k in enumerate(k_values)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-6)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-6)

    def test_newton_vectorized_two_parameters(self, newton_solver):
        """Newton-Raphson vectorized with 2 parameters per solve"""
        n = 10
        a_values = np.ones(n)
        b_values = np.linspace(4.0, 100.0, n)
        func_params = np.column_stack([a_values, b_values])
        x0_array = np.sqrt(b_values) * 0.8

        roots, iters, converged = newton_solver.find_root(
            func_two_params,
            func_two_params_derivative,
            x0=x0_array,
            func_params=func_params,
            tol=1e-6,
        )

        scipy_roots = np.array(
            [
                optimize.newton(
                    lambda x, i=i: a_values[i] * x**2 - b_values[i],
                    x0=x0_array[i],
                    fprime=lambda x, i=i: 2 * a_values[i] * x,
                    tol=1e-6,
                )
                for i in range(n)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-6)

    def test_newton_vectorized_three_parameters(self, newton_solver):
        """Newton-Raphson vectorized with 3 parameters per solve"""
        # Multiple quadratics: a*x^2 + b*x + c = 0
        params = np.array(
            [
                [1.0, -5.0, 6.0],  # x^2 - 5x + 6, roots at 2 and 3
                [1.0, -7.0, 12.0],  # x^2 - 7x + 12, roots at 3 and 4
                [2.0, -8.0, 8.0],  # 2x^2 - 8x + 8, root at 2
            ]
        )
        x0_array = np.array([1.5, 2.5, 1.5])

        roots, iters, converged = newton_solver.find_root(
            func_three_params,
            func_three_params_derivative,
            x0=x0_array,
            func_params=params,
            tol=1e-6,
        )

        scipy_roots = np.array(
            [
                optimize.newton(
                    lambda x, i=i: params[i, 0] * x**2 + params[i, 1] * x + params[i, 2],
                    x0=x0_array[i],
                    fprime=lambda x, i=i: 2 * params[i, 0] * x + params[i, 1],
                    tol=1e-6,
                )
                for i in range(len(params))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-6)

    def test_newton_vectorized_five_parameters(self, newton_solver):
        """Newton-Raphson vectorized with 5 parameters per solve"""
        # x^4 - 10x^2 + 9 has roots at ±1, ±3
        params = np.array(
            [
                [1.0, 0.0, -10.0, 0.0, 9.0],
                [1.0, 0.0, -5.0, 0.0, 4.0],
                [1.0, 0.0, -13.0, 0.0, 12.0],
            ]
        )
        x0_array = np.array([1.5, 1.3, 1.8])

        roots, iters, converged = newton_solver.find_root(
            func_five_params,
            func_five_params_derivative,
            x0=x0_array,
            func_params=params,
            tol=1e-6,
        )

        scipy_roots = np.array(
            [
                optimize.newton(
                    lambda x, i=i: (
                        params[i, 0] * x**4
                        + params[i, 1] * x**3
                        + params[i, 2] * x**2
                        + params[i, 3] * x
                        + params[i, 4]
                    ),
                    x0=x0_array[i],
                    fprime=lambda x, i=i: (
                        4 * params[i, 0] * x**3
                        + 3 * params[i, 1] * x**2
                        + 2 * params[i, 2] * x
                        + params[i, 3]
                    ),
                    tol=1e-6,
                )
                for i in range(len(params))
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-6)

    def test_bisection_vectorized_one_parameter(self, bisection_solver):
        """Bisection vectorized with 1 parameter per solve"""
        k_values = np.array([4.0, 9.0, 16.0, 25.0])
        a_array = np.zeros(4)
        b_array = np.sqrt(k_values) + 2.0

        roots, iters, converged = bisection_solver.find_root(
            func_one_param, a=a_array, b=b_array, func_params=k_values, tol=1e-6
        )

        scipy_roots = np.array(
            [
                optimize.bisect(lambda x, k=k: x**2 - k, a=a_array[i], b=b_array[i], xtol=1e-6)
                for i, k in enumerate(k_values)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-5)

    def test_brent_vectorized_two_parameters(self, brent_solver):
        """Brent vectorized with 2 parameters per solve"""
        n = 20
        a_values = np.ones(n)
        b_values = np.linspace(4.0, 100.0, n)
        func_params = np.column_stack([a_values, b_values])

        a_bounds = np.zeros(n)
        b_bounds = np.sqrt(b_values) + 2.0

        roots, iters, converged = brent_solver.find_root(
            func_two_params, a=a_bounds, b=b_bounds, func_params=func_params, tol=1e-6
        )

        scipy_roots = np.array(
            [
                optimize.brentq(
                    lambda x, i=i: a_values[i] * x**2 - b_values[i],
                    a=a_bounds[i],
                    b=b_bounds[i],
                    xtol=1e-6,
                )
                for i in range(n)
            ]
        )

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-6)


# ============================================================================
# Edge Cases and Robustness
# ============================================================================


class TestEdgeCases:
    """Test solver robustness on challenging cases"""

    def test_newton_steep_function(self, newton_solver):
        """Newton-Raphson on steep function (no params)"""
        root, iters, converged = newton_solver.find_root(
            steep_func, steep_derivative, x0=0.8, tol=1e-6, max_iter=100
        )

        assert converged
        assert np.abs(root - 1.0) < 1e-5

    def test_bisection_tight_bracket(self, bisection_solver):
        """Bisection with very tight initial bracket (no params)"""
        root, iters, converged = bisection_solver.find_root(
            polynomial_func, a=2.09, b=2.10, tol=1e-6
        )

        assert converged
        assert 2.09 <= root <= 2.10

    def test_convergence_flags_scalar(self, newton_solver):
        """Verify convergence flags are correct (scalar)"""
        # Should converge
        root1, iters1, conv1 = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=2.0, max_iter=50
        )
        assert conv1

        # Difficult case with low max_iter (may not converge)
        root2, iters2, conv2 = newton_solver.find_root(
            steep_func, steep_derivative, x0=0.1, max_iter=5
        )
        assert isinstance(conv2, (bool, np.bool_))

    def test_convergence_flags_vectorized(self, newton_solver):
        """Verify convergence flags are correct (vectorized)"""
        x0_array = np.array([2.0, 2.0, 0.1])
        roots, iters, converged = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=x0_array, max_iter=50
        )

        assert isinstance(converged, np.ndarray)
        assert converged.dtype == bool
        assert converged[0] and converged[1]  # These should converge

    def test_mixed_convergence_with_params(self, newton_solver):
        """Test mixed convergence with parameters"""

        # f(x, offset) = x^2 + offset
        # Negative offset: has real roots
        # Positive offset: no real roots
        @njit
        def f(x, offset):
            return x**2 + offset

        @njit
        def fp(x, _offset):
            return 2 * x

        offsets = np.array([-4.0, 1.0, -9.0, 5.0, -16.0])
        x0_array = np.ones(5) * 1.5

        roots, iters, converged = newton_solver.find_root(
            f, fp, x0=x0_array, func_params=offsets, tol=1e-6, max_iter=50
        )

        # Check convergence pattern
        assert converged[0]  # -4 should converge
        assert not converged[1]  # 1 should not converge
        assert converged[2]  # -9 should converge
        assert not converged[3]  # 5 should not converge
        assert converged[4]  # -16 should converge


# ============================================================================
# Large-Scale Performance Tests
# ============================================================================


class TestPerformance:
    """Performance tests on large arrays"""

    def test_newton_large_array_no_params(self, newton_solver):
        """Newton-Raphson on large array (10K elements, no params)"""
        x0_array = np.linspace(1.5, 3.0, 10000)

        roots, iters, converged = newton_solver.find_root(
            polynomial_func, polynomial_derivative, x0=x0_array, tol=1e-6
        )

        assert np.mean(converged) > 0.95
        assert roots.shape == x0_array.shape
        assert iters.shape == x0_array.shape

    def test_newton_large_array_with_params(self, newton_solver):
        """Newton-Raphson on large array (10K elements, with params)"""
        n = 10000
        k_values = np.linspace(1.0, 100.0, n)
        x0_array = np.sqrt(k_values) * 0.8

        roots, iters, converged = newton_solver.find_root(
            func_one_param, func_one_param_derivative, x0=x0_array, func_params=k_values, tol=1e-6
        )

        assert np.all(converged)
        assert roots.shape == (n,)
        assert np.allclose(roots, np.sqrt(k_values), atol=1e-6)

    def test_bisection_large_array_with_params(self, bisection_solver):
        """Bisection on large array (10K elements, with params)"""
        n = 10000
        k_values = np.linspace(1.0, 100.0, n)
        a_array = np.zeros(n)
        b_array = np.sqrt(k_values) + 2.0

        roots, iters, converged = bisection_solver.find_root(
            func_one_param, a=a_array, b=b_array, func_params=k_values, tol=1e-6
        )

        assert np.all(converged)
        assert roots.shape == (n,)

    def test_brent_large_array_two_params(self, brent_solver):
        """Brent on large array (10K elements, 2 params)"""
        n = 10000
        a_values = np.ones(n)
        b_values = np.linspace(1.0, 100.0, n)
        func_params = np.column_stack([a_values, b_values])

        a_bounds = np.zeros(n)
        b_bounds = np.sqrt(b_values) + 2.0

        roots, iters, converged = brent_solver.find_root(
            func_two_params, a=a_bounds, b=b_bounds, func_params=func_params, tol=1e-6
        )

        assert np.all(converged)
        assert roots.shape == (n,)


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.parametrize(
    "func,derivative,x0,expected_root",
    [
        (polynomial_func, polynomial_derivative, 2.0, 2.0946),
        (transcendental_func, transcendental_derivative, 0.5, 0.7391),
        (exponential_func, exponential_derivative, 1.5, 1.512),
    ],
)
def test_newton_multiple_equations_no_params(func, derivative, x0, expected_root):
    """Parametrized test for Newton-Raphson on multiple equations (no params)"""
    solver = NewtonRaphsonSolver()
    root, iters, converged = solver.find_root(func, derivative, x0=x0, tol=1e-4)

    assert converged
    assert np.abs(root - expected_root) < 1e-2


@pytest.mark.parametrize(
    "func,a,b,expected_root",
    [
        (polynomial_func, 2.0, 3.0, 2.0946),
        (transcendental_func, 0.0, 1.0, 0.7391),
        (exponential_func, 0.3, 1.5, 0.6191),
    ],
)
def test_brent_multiple_equations_no_params(func, a, b, expected_root):
    """Parametrized test for Brent on multiple equations (no params)"""
    solver = BrentSolver()
    root, iters, converged = solver.find_root(func, a=a, b=b, tol=1e-4)

    assert converged
    assert np.abs(root - expected_root) < 1e-2


@pytest.mark.parametrize(
    "num_params,k_value",
    [
        (1, 16.0),
        (1, 25.0),
        (1, 36.0),
    ],
)
def test_newton_parametrized_with_params(num_params, k_value):
    """Parametrized test for Newton with parameters"""

    _ = num_params

    solver = NewtonRaphsonSolver()
    root, iters, converged = solver.find_root(
        func_one_param,
        func_one_param_derivative,
        x0=3.0,
        func_params=(k_value,),
        tol=1e-6,
    )

    assert converged
    assert np.isclose(root, np.sqrt(k_value), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
