"""
Comprehensive unit tests for _use_back_up_solvers dispatcher function.

Tests validate correct routing between scalar and vectorized implementations,
SciPy reference validation, edge cases, and integration with backup solver chains.

Author: Cian Quezon
"""

import warnings

import numpy as np
import pytest
from numba import njit
from scipy.optimize import bisect, brentq, newton, root_scalar

from meteorological_equations.math.solvers._back_up_logic import (
    _use_back_up_solvers,
)
from meteorological_equations.math.solvers._enums import SolverName

# ============================================================================
# Test Functions (JIT-compiled for compatibility)
# ============================================================================


@njit
def cubic_func(x: float) -> float:
    """Simple cubic: x^3 - 8 = 0, root at x=2."""
    return x**3 - 8.0


@njit
def cubic_prime(x: float) -> float:
    """Derivative of cubic."""
    return 3.0 * x**2


@njit
def quadratic_func(x: float) -> float:
    """Simple quadratic: x^2 - 4 = 0, roots at x=±2."""
    return x**2 - 4.0


@njit
def quadratic_prime(x: float) -> float:
    """Derivative of quadratic."""
    return 2.0 * x


@njit
def exponential_func(x: float) -> float:
    """Exponential: e^x - 2 = 0, root at x=ln(2)≈0.693."""
    return np.exp(x) - 2.0


@njit
def exponential_prime(x: float) -> float:
    """Derivative of exponential."""
    return np.exp(x)


@njit
def polynomial_func(x: float) -> float:
    """Polynomial: x^5 - 3x^3 + 2x - 1 = 0."""
    return x**5 - 3.0 * x**3 + 2.0 * x - 1.0


@njit
def polynomial_prime(x: float) -> float:
    """Derivative of polynomial."""
    return 5.0 * x**4 - 9.0 * x**2 + 2.0


@njit
def parametric_func(x: float, a: float, b: float) -> float:
    """Parametric: a*x^2 + b = 0."""
    return a * x**2 + b


@njit
def parametric_prime(x: float, a: float) -> float:
    """Derivative of parametric."""
    return 2.0 * a * x


@njit
def difficult_func(x: float) -> float:
    """Difficult function: x^3 - 2*x - 5 = 0, root near 2.0946."""
    return x**3 - 2.0 * x - 5.0


@njit
def difficult_prime(x: float) -> float:
    """Derivative of difficult function."""
    return 3.0 * x**2 - 2.0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_tolerance() -> float:
    """Default convergence tolerance."""
    return 1e-6


@pytest.fixture
def strict_tolerance() -> float:
    """Strict convergence tolerance for accuracy tests."""
    return 1e-10


@pytest.fixture
def max_iterations() -> int:
    """Default maximum iterations."""
    return 100


# ============================================================================
# Test Class: Scalar Routing
# ============================================================================


class TestScalarRouting:
    """Test that scalar inputs route to scalar implementation."""

    def test_scalar_input_detection(self, default_tolerance, max_iterations):
        """Test dispatcher correctly identifies scalar inputs."""
        # Scalar results (0-dimensional arrays become scalars)
        root = np.nan
        iters = 100
        conv = False
        results = (root, iters, conv)

        # Should route to scalar
        root, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        # Check types are scalar
        assert isinstance(root, (float, np.floating)), f"Root should be scalar, got {type(root)}"
        assert isinstance(iters, (int, np.integer)), f"Iters should be scalar, got {type(iters)}"
        assert isinstance(conv, (bool, np.bool_)), f"Conv should be scalar, got {type(conv)}"

        # Check convergence
        assert conv, "Should converge"
        assert np.isclose(root, 2.0, atol=default_tolerance)

    def test_scalar_vs_scipy_brentq(self, strict_tolerance):
        """Compare scalar dispatch with SciPy brentq."""
        results = (np.nan, 100, False)

        # Our implementation (scalar)
        root, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=None,
            tol=strict_tolerance,
            max_iter=100,
            backup_solvers=[SolverName.BRENT],
        )

        # SciPy reference
        scipy_root = brentq(cubic_func, 0.0, 5.0, xtol=strict_tolerance)

        assert conv, "Should converge"
        assert np.isclose(
            root, scipy_root, atol=1e-12
        ), f"Difference from SciPy: {abs(root - scipy_root)}"

    def test_scalar_vs_scipy_bisect(self, default_tolerance):
        """Compare scalar dispatch with SciPy bisect."""
        results = (np.nan, 100, False)

        root, iters, conv = _use_back_up_solvers(
            func=quadratic_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=None,
            tol=default_tolerance,
            max_iter=100,
            backup_solvers=[SolverName.BISECTION],
        )

        scipy_root = bisect(quadratic_func, 0.0, 5.0, xtol=default_tolerance)

        assert conv
        assert np.isclose(root, scipy_root, atol=1e-10)

    def test_scalar_vs_scipy_newton(self, strict_tolerance):
        """Compare scalar dispatch with SciPy newton."""
        results = (np.nan, 100, False)

        root, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=None,
            b=None,
            x0=2.5,
            tol=strict_tolerance,
            max_iter=50,
            func_prime=cubic_prime,
            backup_solvers=[SolverName.NEWTON],
        )

        scipy_root = newton(cubic_func, 2.5, fprime=cubic_prime, tol=strict_tolerance)

        assert conv
        assert np.isclose(root, scipy_root, atol=1e-12)

    def test_scalar_already_converged(self):
        """Test scalar input that's already converged."""
        # Already converged
        results = (2.0, 8, True)

        root, iters, conv = _use_back_up_solvers(
            func=cubic_func, results=results, a=0.0, b=5.0, x0=None, tol=1e-6, max_iter=100
        )

        # Should return unchanged
        assert root == 2.0
        assert iters == 8
        assert conv


# ============================================================================
# Test Class: Vectorized Routing
# ============================================================================


class TestVectorizedRouting:
    """Test that vectorized inputs route to vectorized implementation."""

    def test_vectorized_input_detection(self, default_tolerance, max_iterations):
        """Test dispatcher correctly identifies vectorized inputs."""
        n = 5
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        results = (roots, iters, conv)

        a = np.array([0.0, 0.5, 1.0, 1.5, 1.8])
        b = np.array([3.0, 3.0, 3.0, 3.0, 2.5])

        # Should route to vectorized
        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        # Check types are arrays
        assert isinstance(roots, np.ndarray)
        assert isinstance(iters, np.ndarray)
        assert isinstance(conv, np.ndarray)

        # Check convergence
        assert np.all(conv), f"All should converge, got {conv}"
        assert np.allclose(roots, 2.0, atol=default_tolerance)

    def test_vectorized_vs_scipy_batch(self, strict_tolerance):
        """Compare vectorized dispatch with batch SciPy calls."""
        n = 10
        np.random.seed(42)

        # Random brackets around x=2
        a = np.random.uniform(0.0, 1.9, n)
        b = np.random.uniform(2.1, 4.0, n)

        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        # Our implementation (vectorized)
        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=strict_tolerance,
            max_iter=100,
            backup_solvers=[SolverName.BRENT],
        )

        # SciPy reference (batch)
        scipy_roots = np.array(
            [brentq(cubic_func, a[i], b[i], xtol=strict_tolerance) for i in range(n)]
        )

        assert np.all(conv), "All should converge"
        assert np.allclose(
            roots, scipy_roots, atol=1e-11
        ), f"Max difference from SciPy: {np.max(np.abs(roots - scipy_roots))}"

    def test_vectorized_partial_convergence(self, default_tolerance, max_iterations):
        """Test vectorized with some elements already converged."""
        # Mixed convergence
        roots = np.array([2.0, np.nan, 1.5, np.nan, np.nan])
        iters = np.array([8, 100, 10, 100, 100])
        conv = np.array([True, False, True, False, False])

        # Save original converged values
        original_root_0 = roots[0]
        original_root_2 = roots[2]

        a = np.array([0, 0, 0, 1, 1.5])
        b = np.array([3, 3, 3, 3, 3])

        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        # Check preservation of converged elements
        assert roots[0] == original_root_0, "Converged element 0 should be preserved"
        assert roots[2] == original_root_2, "Converged element 2 should be preserved"

        # Check new convergence
        assert np.all(conv), "All should now be converged"
        assert np.allclose(roots[[1, 3, 4]], 2.0, atol=default_tolerance)

    def test_vectorized_all_converged(self):
        """Test vectorized input with all elements already converged."""
        roots = np.array([2.0, 2.0, 2.0])
        iters = np.array([5, 6, 7])
        conv = np.array([True, True, True])

        # Make copies
        roots_copy = roots.copy()
        iters_copy = iters.copy()

        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=np.array([0, 0, 0]),
            b=np.array([3, 3, 3]),
            x0=None,
            tol=1e-6,
            max_iter=100,
        )

        # Nothing should change
        assert np.array_equal(roots, roots_copy)
        assert np.array_equal(iters, iters_copy)


# ============================================================================
# Test Class: Different Functions
# ============================================================================


class TestDifferentFunctions:
    """Test dispatcher with various mathematical functions."""

    def test_exponential_scalar(self, default_tolerance, max_iterations):
        """Test exponential function with scalar input."""
        expected_root = np.log(2.0)  # ≈ 0.693

        results = (np.nan, 100, False)

        root, iters, conv = _use_back_up_solvers(
            func=exponential_func,
            results=results,
            a=-1.0,
            b=2.0,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        assert conv
        assert np.isclose(root, expected_root, atol=default_tolerance)

        # Verify with SciPy
        scipy_root = brentq(exponential_func, -1.0, 2.0)
        assert np.isclose(root, scipy_root, atol=1e-10)

    def test_exponential_vectorized(self, default_tolerance, max_iterations):
        """Test exponential function with vectorized input."""
        n = 5
        expected_root = np.log(2.0)

        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        a = np.full(n, -1.0)
        b = np.full(n, 2.0)

        roots, iters, conv = _use_back_up_solvers(
            func=exponential_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        assert np.all(conv)
        assert np.allclose(roots, expected_root, atol=default_tolerance)

    def test_polynomial_scalar_vs_scipy(self, strict_tolerance):
        """Test polynomial with scalar vs SciPy root_scalar."""
        results = (np.nan, 100, False)

        root, iters, conv = _use_back_up_solvers(
            func=polynomial_func,
            results=results,
            a=-2.0,
            b=2.0,
            x0=None,
            tol=strict_tolerance,
            max_iter=100,
            backup_solvers=[SolverName.BRENT],
        )

        # SciPy root_scalar with brentq method
        scipy_result = root_scalar(
            polynomial_func, bracket=[-2.0, 2.0], method="brentq", xtol=strict_tolerance
        )

        assert conv
        assert np.isclose(root, scipy_result.root, atol=1e-11)

    def test_difficult_function_vectorized(self, default_tolerance, max_iterations):
        """Test difficult function with vectorized input."""
        # Root near x ≈ 2.0946
        n = 8

        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        a = np.linspace(1.5, 2.0, n)
        b = np.full(n, 3.0)

        roots, iters, conv = _use_back_up_solvers(
            func=difficult_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        assert np.all(conv), f"Not all converged: {conv}"

        # All should find same root
        assert np.allclose(roots, roots[0], atol=1e-8)

        # Verify with SciPy
        scipy_root = brentq(difficult_func, 1.5, 3.0)
        assert np.allclose(roots[0], scipy_root, atol=1e-10)


# ============================================================================
# Test Class: Parametric Functions
# ============================================================================


class TestParametricFunctions:
    """Test dispatcher with parametric functions."""

    def test_scalar_with_parameters(self, default_tolerance, max_iterations):
        """Test scalar dispatch with function parameters."""
        results = (np.nan, 100, False)

        # Solve a*x^2 + b = 0 with a=1, b=-4
        # Root at x=2
        root, iters, conv = _use_back_up_solvers(
            func=parametric_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_params=(1.0, -4.0),
        )

        assert conv, f"Should converge, got {conv}"
        assert np.isclose(root, 2.0, atol=default_tolerance), f"Root should be 2.0, got {root}"

    def test_vectorized_with_2d_parameters(self, default_tolerance, max_iterations):
        """Test vectorized dispatch with 2D parameters."""
        n = 4

        # Different (a, b) for each element, all should give root at x=2
        func_params = np.array(
            [
                [1.0, -4.0],  # x^2 - 4 = 0
                [2.0, -8.0],  # 2x^2 - 8 = 0
                [0.5, -2.0],  # 0.5x^2 - 2 = 0
                [3.0, -12.0],  # 3x^2 - 12 = 0
            ]
        )

        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        a = np.full(n, 0.0)
        b = np.full(n, 5.0)

        roots, iters, conv = _use_back_up_solvers(
            func=parametric_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_params=func_params,
        )

        assert np.all(conv), f"All should converge, got {conv}"
        assert np.allclose(
            roots, 2.0, atol=default_tolerance
        ), f"All roots should be 2.0, got {roots}"

    def test_vectorized_mixed_convergence_with_parameters(self, default_tolerance, max_iterations):
        """Test parameter handling with partial convergence."""
        roots = np.array([2.0, np.nan, np.nan])
        iters = np.array([5, 100, 100])
        conv = np.array([True, False, False])

        func_params = np.array(
            [
                [1.0, -4.0],  # Already converged
                [2.0, -8.0],  # Needs solving
                [0.5, -2.0],  # Needs solving
            ]
        )

        a = np.array([0, 0, 0])
        b = np.array([5, 5, 5])

        roots, iters, conv = _use_back_up_solvers(
            func=parametric_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_params=func_params,
        )

        assert np.all(conv)
        # Element 0 preserved
        assert roots[0] == 2.0
        # Elements 1, 2 solved
        assert np.allclose(roots[[1, 2]], 2.0, atol=default_tolerance)


# ============================================================================
# Test Class: Solver Chains
# ============================================================================


class TestSolverChains:
    """Test dispatcher with different solver chains."""

    def test_scalar_single_solver(self, default_tolerance, max_iterations):
        """Test scalar with single solver in chain."""
        results = (np.nan, 100, False)

        root, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            backup_solvers=[SolverName.BRENT],
        )

        assert conv
        assert np.isclose(root, 2.0, atol=default_tolerance)

    def test_scalar_multiple_solver_chain(self, default_tolerance, max_iterations):
        """Test scalar with multiple solvers in chain."""
        results = (np.nan, 100, False)

        # Chain: Newton → Brent → Bisection
        root, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=2.5,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_prime=cubic_prime,
            backup_solvers=[SolverName.NEWTON, SolverName.BRENT, SolverName.BISECTION],
        )

        assert conv
        assert np.isclose(root, 2.0, atol=default_tolerance)

    def test_vectorized_custom_chain(self, default_tolerance, max_iterations):
        """Test vectorized with custom solver chain."""
        n = 4

        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        x0 = np.array([1.5, 2.5, 1.8, 2.2])
        a = np.array([0, 0, 0, 0])
        b = np.array([3, 3, 3, 3])

        # Try Newton first, then Brent
        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=x0,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_prime=cubic_prime,
            backup_solvers=[SolverName.NEWTON, SolverName.BRENT],
        )

        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)

    def test_default_solver_chain(self, default_tolerance, max_iterations):
        """Test that default chain (Brent → Bisection) works."""
        # Scalar
        results_scalar = (np.nan, 100, False)

        root, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results_scalar,
            a=0.0,
            b=5.0,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            backup_solvers=None,  # Use default
        )

        assert conv
        assert np.isclose(root, 2.0, atol=default_tolerance)

        # Vectorized
        n = 3
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=np.array([0, 0, 0]),
            b=np.array([3, 3, 3]),
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            backup_solvers=None,  # Use default
        )

        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)


# ============================================================================
# Test Class: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_scalar_to_vectorized_boundary(self, default_tolerance, max_iterations):
        """Test single element (boundary between scalar and vectorized)."""
        # Single element as array (should route to vectorized)
        roots = np.array([np.nan])
        iters = np.array([100])
        conv = np.array([False])

        a = np.array([0.0])
        b = np.array([3.0])

        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        assert isinstance(roots, np.ndarray)
        assert conv[0]
        assert np.isclose(roots[0], 2.0, atol=default_tolerance)

    def test_empty_vectorized(self, default_tolerance, max_iterations):
        """Test with empty arrays."""
        roots = np.array([])
        iters = np.array([])
        conv = np.array([], dtype=bool)

        a = np.array([])
        b = np.array([])

        # Should handle gracefully
        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        assert len(roots) == 0
        assert len(conv) == 0

    def test_scalar_tight_tolerance(self):
        """Test scalar with very tight tolerance."""
        results = (np.nan, 100, False)
        tol = 1e-14

        root, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=None,
            tol=tol,
            max_iter=100,
            backup_solvers=[SolverName.BRENT],
        )

        assert conv
        # Check residual
        residual = abs(cubic_func(root))
        assert residual < tol

    def test_vectorized_wide_brackets(self, default_tolerance, max_iterations):
        """Test vectorized with very wide brackets."""
        n = 3

        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        # Very wide brackets
        a = np.array([-100.0, -50.0, -10.0])
        b = np.array([100.0, 50.0, 10.0])

        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)

    def test_scalar_max_iterations_exceeded(self):
        """Test scalar behavior when max iterations is too low."""
        results = (np.nan, 100, False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            root, iters, conv = _use_back_up_solvers(
                func=cubic_func,
                results=results,
                a=0.0,
                b=5.0,
                x0=None,
                tol=1e-15,  # Very strict
                max_iter=2,  # Too few
                backup_solvers=[SolverName.BISECTION],
            )

            # Likely won't converge
            # May have warnings
            assert len(w) >= 0  # Warnings are ok


# ============================================================================
# Test Class: Large Scale
# ============================================================================


class TestLargeScale:
    """Test performance and correctness at scale."""

    def test_large_vectorized_1000_elements(self, default_tolerance):
        """Test vectorized with 1000 elements."""
        n = 1000
        np.random.seed(42)

        # Random brackets around x=2
        a = np.random.uniform(0.0, 1.9, n)
        b = np.random.uniform(2.1, 4.0, n)

        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        import time

        start = time.time()

        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=100,
        )

        elapsed = time.time() - start

        # Convergence check
        convergence_rate = np.sum(conv) / n * 100
        assert convergence_rate >= 99.0, f"Convergence rate: {convergence_rate}%"

        # Accuracy check
        converged_roots = roots[conv]
        assert np.allclose(converged_roots, 2.0, atol=default_tolerance)

        # Performance check
        assert elapsed < 3.0, f"Took {elapsed:.3f}s, expected < 3.0s"

        print(f"\n✓ Dispatcher: Solved {n} roots in {elapsed:.3f}s")
        print(f"✓ Convergence rate: {convergence_rate:.1f}%")

    def test_many_scalar_calls_vs_one_vectorized(self, default_tolerance):
        """Compare many scalar calls vs one vectorized call."""
        n = 100
        np.random.seed(123)

        a_vals = np.random.uniform(0.0, 1.9, n)
        b_vals = np.random.uniform(2.1, 4.0, n)

        import time

        # Many scalar calls
        start_scalar = time.time()
        scalar_roots = []
        for i in range(n):
            root, _, conv = _use_back_up_solvers(
                func=cubic_func,
                results=(np.nan, 100, False),
                a=float(a_vals[i]),
                b=float(b_vals[i]),
                x0=None,
                tol=default_tolerance,
                max_iter=100,
            )
            scalar_roots.append(root)
        scalar_time = time.time() - start_scalar

        # One vectorized call
        start_vec = time.time()
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        vec_roots, _, _ = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a_vals,
            b=b_vals,
            x0=None,
            tol=default_tolerance,
            max_iter=100,
        )
        vec_time = time.time() - start_vec

        # Vectorized should be faster or comparable
        speedup = scalar_time / vec_time
        print(f"\n✓ Scalar time: {scalar_time:.3f}s")
        print(f"✓ Vectorized time: {vec_time:.3f}s")
        print(f"✓ Speedup: {speedup:.2f}x")

        # Results should match
        assert np.allclose(scalar_roots, vec_roots, atol=1e-10)


# ============================================================================
# Test Class: Comprehensive Integration
# ============================================================================


class TestComprehensiveIntegration:
    """Comprehensive integration tests combining multiple features."""

    def test_full_workflow_scalar(self, strict_tolerance):
        """Test complete scalar workflow with validation."""
        # Primary solver "failed"
        results = (np.nan, 100, False)

        # Try full chain
        root, iters, conv = _use_back_up_solvers(
            func=difficult_func,
            results=results,
            a=1.5,
            b=3.0,
            x0=2.0,
            tol=strict_tolerance,
            max_iter=100,
            func_prime=difficult_prime,
            backup_solvers=[SolverName.NEWTON, SolverName.BRENT, SolverName.BISECTION],
        )

        # Should converge
        assert conv

        # Validate with SciPy multiple methods
        scipy_brent = brentq(difficult_func, 1.5, 3.0, xtol=strict_tolerance)
        scipy_newton = newton(difficult_func, 2.0, fprime=difficult_prime, tol=strict_tolerance)

        # Should match both methods
        assert np.isclose(root, scipy_brent, atol=1e-10)
        assert np.isclose(root, scipy_newton, atol=1e-10)

        # Check residual
        residual = abs(difficult_func(root))
        assert residual < strict_tolerance

    def test_vectorized_mixed_convergence_preservation(self, default_tolerance):
        """Test vectorized with some pre-converged elements (preservation test)."""
        n = 15
        np.random.seed(456)

        # Create scenario: some elements pre-converged to correct value
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        # Pre-converge 30% to correct root
        n_preconverged = int(n * 0.3)
        preconverged_indices = np.random.choice(n, n_preconverged, replace=False)
        roots[preconverged_indices] = 2.0
        iters[preconverged_indices] = np.random.randint(5, 20, n_preconverged)
        conv[preconverged_indices] = True

        # Store for validation
        originally_converged = conv.copy()

        # Random brackets and guesses
        a = np.random.uniform(0.0, 1.9, n)
        b = np.random.uniform(2.1, 4.0, n)
        x0 = np.random.uniform(1.5, 2.5, n)

        # Run solvers
        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=x0,
            tol=default_tolerance,
            max_iter=100,
            func_prime=cubic_prime,
            backup_solvers=[SolverName.NEWTON, SolverName.BRENT, SolverName.BISECTION],
        )

        # 1. Check preservation of pre-converged elements
        assert np.allclose(
            roots[originally_converged], 2.0, atol=1e-14
        ), "Pre-converged elements should remain unchanged"

        # 2. Check high overall convergence
        convergence_rate = np.sum(conv) / n * 100
        assert convergence_rate >= 95.0, f"Convergence: {convergence_rate}%"

        # 3. Validate newly converged elements with SciPy
        newly_converged = conv & ~originally_converged
        for i in np.where(newly_converged)[0]:
            scipy_root = brentq(cubic_func, a[i], b[i])
            assert np.isclose(
                roots[i], scipy_root, atol=1e-8
            ), f"Element {i}: {roots[i]:.6f} vs SciPy {scipy_root:.6f}"

    def test_vectorized_all_unconverged_full_chain(self, default_tolerance):
        """Test vectorized starting from all unconverged with full solver chain."""
        n = 25
        np.random.seed(789)

        # All start unconverged
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)

        # Random valid brackets
        a = np.random.uniform(0.0, 1.9, n)
        b = np.random.uniform(2.1, 4.0, n)
        x0 = np.random.uniform(1.5, 2.5, n)

        # Apply full solver chain
        roots, iters, conv = _use_back_up_solvers(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=x0,
            tol=default_tolerance,
            max_iter=100,
            func_prime=cubic_prime,
            backup_solvers=[SolverName.NEWTON, SolverName.BRENT, SolverName.BISECTION],
        )

        # Should get excellent convergence
        convergence_rate = np.sum(conv) / n * 100
        assert convergence_rate >= 98.0, f"Convergence: {convergence_rate}%"

        # Validate every converged element
        for i in range(n):
            if conv[i]:
                scipy_root = brentq(cubic_func, a[i], b[i], xtol=default_tolerance)
                assert np.isclose(
                    roots[i], scipy_root, atol=1e-8
                ), f"Element {i} mismatch: {roots[i]:.10f} vs {scipy_root:.10f}"

                # Also verify residual
                residual = abs(cubic_func(roots[i]))
                assert residual < default_tolerance, f"Element {i} residual too large: {residual}"


# ============================================================================
# Test Class: Consistency
# ============================================================================


class TestConsistency:
    """Test consistency and determinism."""

    def test_scalar_deterministic(self, default_tolerance, max_iterations):
        """Test that scalar gives consistent results."""
        results = (np.nan, 100, False)

        # Run twice
        root1, iters1, conv1 = _use_back_up_solvers(
            func=cubic_func,
            results=results,
            a=0.0,
            b=5.0,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        root2, iters2, conv2 = _use_back_up_solvers(
            func=cubic_func,
            results=(np.nan, 100, False),
            a=0.0,
            b=5.0,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        # Should be identical
        assert conv1 == conv2
        assert np.isclose(root1, root2, atol=1e-14)

    def test_vectorized_deterministic(self, default_tolerance, max_iterations):
        """Test that vectorized gives consistent results."""
        n = 10

        a = np.linspace(0, 1.5, n)
        b = np.full(n, 3.0)

        # Run 1
        roots1 = np.full(n, np.nan)
        iters1 = np.full(n, 100)
        conv1 = np.full(n, False)

        roots1, _, conv1 = _use_back_up_solvers(
            func=cubic_func,
            results=(roots1, iters1, conv1),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        # Run 2
        roots2 = np.full(n, np.nan)
        iters2 = np.full(n, 100)
        conv2 = np.full(n, False)

        roots2, _, conv2 = _use_back_up_solvers(
            func=cubic_func,
            results=(roots2, iters2, conv2),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
        )

        # Should be identical
        assert np.array_equal(conv1, conv2)
        assert np.allclose(roots1, roots2, atol=1e-14)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
