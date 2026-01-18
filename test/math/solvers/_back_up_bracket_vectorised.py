"""
Comprehensive tests for bracket backup dispatcher.

Tests the _try_back_up_bracket_vectorised function against SciPy
to ensure correctness, precision, and robustness.

Author: Test Suite
"""

import warnings

import numpy as np
import pytest
from numba import njit
from scipy import optimize

from meteorological_equations.math.solvers._back_up_logic import (
    _try_back_up_bracket_vectorised,
)
from meteorological_equations.math.solvers._solvers import (
    BisectionSolver,
    BrentSolver,
)

# =============================================================================
# TEST FUNCTIONS (Numba-compatible, single real roots)
# =============================================================================


@njit
def cubic_minus_8(x):
    """x^3 - 8 = 0, single real root at x=2"""
    return x**3 - 8


@njit
def exponential_minus_e(x):
    """exp(x) - e = 0, single root at x=1"""
    return np.exp(x) - np.e


@njit
def sin_function(x):
    """sin(x) = 0 in [3, 4], root at π"""
    return np.sin(x)


@njit
def polynomial_quartic(x):
    """(x-3)^4 - 1 = 0, has root at x=2 and x=4"""
    return (x - 3) ** 4 - 1


@njit
def log_function(x):
    """ln(x) - 1 = 0, root at x=e"""
    return np.log(x) - 1


@njit
def transcendental(x):
    """x*exp(x) - 1 = 0"""
    return x * np.exp(x) - 1


# Non-JIT versions for SciPy
def cubic_minus_8_scipy(x):
    return x**3 - 8


def exponential_minus_e_scipy(x):
    return np.exp(x) - np.e


def sin_function_scipy(x):
    return np.sin(x)


def log_function_scipy(x):
    return np.log(x) - 1


def transcendental_scipy(x):
    return x * np.exp(x) - 1


# =============================================================================
# TEST CLASS: Bracket Vectorised Dispatcher
# =============================================================================


class TestTryBackupBracketVectorised:
    """Comprehensive tests for _try_back_up_bracket_vectorised"""

    def test_basic_functionality_all_unconverged(self):
        """Test basic case where all elements are unconverged"""
        # Setup: 5 unconverged elements
        roots = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100, 100, 100])
        converged = np.array([False, False, False, False, False])
        results = (roots, iterations, converged)

        # Brackets for x^3 - 8 = 0 (root at x=2)
        a = np.array([0.0, 0.5, 1.0, 1.5, 1.8])
        b = np.array([3.0, 3.0, 3.0, 3.0, 2.5])
        unconverged_idx = np.array([0, 1, 2, 3, 4])

        solver = BrentSolver()

        # Run backup
        success = _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a,
            b=b,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
        )

        # Verify
        assert success, "Should return True if any converged"
        assert np.all(converged), "All should have converged"
        assert np.allclose(roots, 2.0, atol=1e-5), f"All roots should be near 2.0, got {roots}"

    def test_partial_convergence(self):
        """Test case where only some elements converge"""
        # Setup: 5 elements, 3 unconverged
        roots = np.array([2.0, np.nan, 1.5, np.nan, np.nan])
        iterations = np.array([10, 100, 8, 100, 100])
        converged = np.array([True, False, True, False, False])
        results = (roots, iterations, converged)

        # Only provide brackets for unconverged indices
        a = np.array([0.0, 0.0, 0.0, 1.0, 1.5])
        b = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        unconverged_idx = np.array([1, 3, 4])  # Only these need solving

        solver = BrentSolver()

        # Run backup on unconverged only
        success = _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a,
            b=b,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
        )

        # Verify
        assert success

        # Already converged should be unchanged
        assert roots[0] == 2.0
        assert roots[2] == 1.5
        assert converged[0]
        assert converged[2]

        # Previously unconverged should now be converged
        assert converged[1]
        assert converged[3]
        assert converged[4]
        assert np.allclose(roots[1], 2.0, atol=1e-5)
        assert np.allclose(roots[3], 2.0, atol=1e-5)
        assert np.allclose(roots[4], 2.0, atol=1e-5)

    def test_precision_vs_scipy_brentq(self):
        """Test numerical precision matches SciPy's brentq"""
        # Setup
        roots = np.array([np.nan, np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100, 100])
        converged = np.array([False, False, False, False])
        results = (roots, iterations, converged)

        # Different brackets
        a_arr = np.array([0.5, 0.8, 1.2, 1.5])
        b_arr = np.array([3.0, 2.8, 2.5, 2.2])
        unconverged_idx = np.array([0, 1, 2, 3])

        solver = BrentSolver()

        # Your implementation
        _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a_arr,
            b=b_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-10,
            max_iter=100,
            func_params=None,
        )

        # SciPy reference (brentq)
        scipy_roots = np.array(
            [optimize.brentq(cubic_minus_8_scipy, a, b, xtol=1e-10) for a, b in zip(a_arr, b_arr)]
        )

        # Compare precision
        print(f"\nYour roots:  {roots}")
        print(f"SciPy roots: {scipy_roots}")
        print(f"Difference:  {np.abs(roots - scipy_roots)}")

        assert np.all(converged), "All should converge"
        assert np.allclose(roots, scipy_roots, atol=1e-9), (
            f"Precision mismatch: yours={roots}, scipy={scipy_roots}"
        )

    def test_precision_vs_scipy_bisect(self):
        """Test numerical precision matches SciPy's bisect"""
        # Setup
        roots = np.array([np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100])
        converged = np.array([False, False, False])
        results = (roots, iterations, converged)

        a_arr = np.array([0.0, 0.5, 1.0])
        b_arr = np.array([3.0, 3.0, 3.0])
        unconverged_idx = np.array([0, 1, 2])

        solver = BisectionSolver()

        # Your implementation
        _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a_arr,
            b=b_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-8,
            max_iter=100,
            func_params=None,
        )

        # SciPy reference (bisect)
        scipy_roots = np.array(
            [optimize.bisect(cubic_minus_8_scipy, a, b, xtol=1e-8) for a, b in zip(a_arr, b_arr)]
        )

        print("\nBisection:")
        print(f"  Your roots:  {roots}")
        print(f"  SciPy roots: {scipy_roots}")
        print(f"  Difference:  {np.abs(roots - scipy_roots)}")

        assert np.all(converged)
        assert np.allclose(roots, scipy_roots, atol=1e-7)

    def test_multiple_functions(self):
        """Test with different mathematical functions"""
        test_cases = [
            (cubic_minus_8, np.array([0.0, 0.5, 1.0]), np.array([3.0, 3.0, 3.0]), 2.0),
            (exponential_minus_e, np.array([0.0, 0.5, 0.8]), np.array([2.0, 2.0, 1.5]), 1.0),
            (sin_function, np.array([3.0, 3.1, 3.05]), np.array([3.5, 3.3, 3.2]), np.pi),
            (log_function, np.array([1.0, 1.5, 2.0]), np.array([4.0, 4.0, 4.0]), np.e),
        ]

        for func, a_arr, b_arr, expected_root in test_cases:
            roots = np.full(len(a_arr), np.nan)
            iterations = np.full(len(a_arr), 100)
            converged = np.full(len(a_arr), False)
            results = (roots, iterations, converged)

            unconverged_idx = np.arange(len(a_arr))
            solver = BrentSolver()

            success = _try_back_up_bracket_vectorised(
                backup_solver=solver,
                func=func,
                results=results,
                a=a_arr,
                b=b_arr,
                unconverged_idx=unconverged_idx,
                tol=1e-8,
                max_iter=100,
                func_params=None,
            )

            assert success, f"Failed for function {func.__name__}"
            assert np.all(converged), f"Not all converged for {func.__name__}"
            assert np.allclose(roots, expected_root, atol=1e-6), (
                f"Expected {expected_root}, got {roots} for {func.__name__}"
            )

    def test_with_func_params(self):
        """Test with function parameters"""

        @njit
        def parametric_func(x, a, b, c):
            """a*x^3 + b*x + c = 0"""
            return a * x**3 + b * x + c

        # Setup: a=1, b=0, c=-8 gives x^3 - 8 = 0 (root at x=2)
        roots = np.array([np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100])
        converged = np.array([False, False, False])
        results = (roots, iterations, converged)

        a_arr = np.array([0.0, 0.5, 1.0])
        b_arr = np.array([3.0, 3.0, 3.0])
        unconverged_idx = np.array([0, 1, 2])

        # Parameters: [a=1, b=0, c=-8] for each element
        func_params = np.array([[1.0, 0.0, -8.0], [1.0, 0.0, -8.0], [1.0, 0.0, -8.0]])

        solver = BrentSolver()

        success = _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=parametric_func,
            results=results,
            a=a_arr,
            b=b_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=func_params,
        )

        assert success
        assert np.all(converged)
        assert np.allclose(roots, 2.0, atol=1e-5)

    def test_invalid_brackets_returns_false(self):
        """Test that invalid brackets (f(a)*f(b) > 0) are handled"""

        # Setup with INVALID brackets (both positive)
        roots = np.array([np.nan, np.nan])
        iterations = np.array([100, 100])
        converged = np.array([False, False])
        results = (roots, iterations, converged)

        # Invalid: f(3) and f(4) both positive for x^3 - 8
        a_arr = np.array([3.0, 3.0])
        b_arr = np.array([4.0, 5.0])
        unconverged_idx = np.array([0, 1])

        solver = BrentSolver()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            success = _try_back_up_bracket_vectorised(
                backup_solver=solver,
                func=cubic_minus_8,
                results=results,
                a=a_arr,
                b=b_arr,
                unconverged_idx=unconverged_idx,
                tol=1e-6,
                max_iter=100,
                func_params=None,
            )

        # Should return False (failure)
        assert not success, "Should return False for invalid brackets"
        assert not np.all(converged), "Should not all converge with invalid brackets"

    def test_empty_unconverged_idx(self):
        """Test with empty unconverged indices"""
        # Setup: all already converged
        roots = np.array([2.0, 2.0, 2.0])
        iterations = np.array([10, 10, 10])
        converged = np.array([True, True, True])
        results = (roots, iterations, converged)

        a_arr = np.array([0.0, 0.0, 0.0])
        b_arr = np.array([3.0, 3.0, 3.0])
        unconverged_idx = np.array([])  # Empty!

        solver = BrentSolver()

        # Should handle gracefully
        success = _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a_arr,
            b=b_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
        )

        # Nothing to do, but should not crash
        assert isinstance(success, (bool, np.bool_))
        # Results unchanged
        assert np.all(roots == 2.0)
        assert np.all(converged)

    def test_iterations_updated(self):
        """Test that iteration counts are properly updated"""
        roots = np.array([np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100])  # Placeholder
        converged = np.array([False, False, False])
        results = (roots, iterations, converged)

        a_arr = np.array([0.0, 0.5, 1.0])
        b_arr = np.array([3.0, 3.0, 3.0])
        unconverged_idx = np.array([0, 1, 2])

        solver = BrentSolver()

        _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a_arr,
            b=b_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
        )

        # Iterations should be updated (not 100 anymore)
        assert np.all(iterations != 100), "Iterations should be updated"
        assert np.all(iterations > 0), "Should have positive iterations"
        assert np.all(iterations < 50), "Brent should converge quickly"

    def test_different_bracket_solvers(self):
        """Test with different bracket-based solvers"""
        solvers = [
            (BrentSolver(), "Brent"),
            (BisectionSolver(), "Bisection"),
        ]

        for solver, name in solvers:
            roots = np.array([np.nan, np.nan, np.nan])
            iterations = np.array([100, 100, 100])
            converged = np.array([False, False, False])
            results = (roots, iterations, converged)

            a_arr = np.array([0.0, 0.5, 1.0])
            b_arr = np.array([3.0, 3.0, 3.0])
            unconverged_idx = np.array([0, 1, 2])

            success = _try_back_up_bracket_vectorised(
                backup_solver=solver,
                func=cubic_minus_8,
                results=results,
                a=a_arr,
                b=b_arr,
                unconverged_idx=unconverged_idx,
                tol=1e-6,
                max_iter=100,
                func_params=None,
            )

            assert success, f"{name} should succeed"
            assert np.all(converged), f"All should converge with {name}"
            assert np.allclose(roots, 2.0, atol=1e-5), (
                f"{name} should find root at 2.0, got {roots}"
            )

    def test_large_array(self):
        """Test with larger array (100 elements)"""
        n = 100
        roots = np.full(n, np.nan)
        iterations = np.full(n, 100)
        converged = np.full(n, False)
        results = (roots, iterations, converged)

        # Different brackets, all containing x=2
        np.random.seed(42)
        a_arr = np.random.uniform(0.0, 1.9, n)
        b_arr = np.random.uniform(2.1, 4.0, n)
        unconverged_idx = np.arange(n)

        solver = BrentSolver()

        success = _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a_arr,
            b=b_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
        )

        assert success
        assert np.all(converged), f"Expected all to converge, got {np.sum(converged)}/{n}"
        assert np.allclose(roots, 2.0, atol=1e-5)

    def test_narrow_brackets(self):
        """Test with very narrow brackets"""
        roots = np.array([np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100])
        converged = np.array([False, False, False])
        results = (roots, iterations, converged)

        # Very narrow brackets around x=2
        a_arr = np.array([1.95, 1.98, 1.99])
        b_arr = np.array([2.05, 2.02, 2.01])
        unconverged_idx = np.array([0, 1, 2])

        solver = BrentSolver()

        success = _try_back_up_bracket_vectorised(
            backup_solver=solver,
            func=cubic_minus_8,
            results=results,
            a=a_arr,
            b=b_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-10,
            max_iter=100,
            func_params=None,
        )

        assert success
        assert np.all(converged)
        assert np.allclose(roots, 2.0, atol=1e-9)

    def test_comprehensive_scipy_comparison(self):
        """Comprehensive comparison with SciPy across multiple scenarios"""

        test_scenarios = [
            (
                "Cubic",
                cubic_minus_8,
                cubic_minus_8_scipy,
                np.array([0.0, 0.5, 1.0, 1.5]),
                np.array([3.0, 3.0, 3.0, 3.0]),
                2.0,
            ),
            (
                "Exponential",
                exponential_minus_e,
                exponential_minus_e_scipy,
                np.array([0.0, 0.3, 0.5, 0.8]),
                np.array([2.0, 2.0, 2.0, 1.5]),
                1.0,
            ),
            (
                "Sine",
                sin_function,
                sin_function_scipy,
                np.array([3.0, 3.05, 3.1, 3.12]),
                np.array([3.5, 3.3, 3.2, 3.18]),
                np.pi,
            ),
            (
                "Logarithm",
                log_function,
                log_function_scipy,
                np.array([1.0, 1.5, 2.0, 2.5]),
                np.array([4.0, 4.0, 4.0, 4.0]),
                np.e,
            ),
        ]

        print("\n" + "=" * 80)
        print("COMPREHENSIVE SCIPY COMPARISON (BRACKET METHODS)")
        print("=" * 80)

        for name, func_jit, func_scipy, a_arr, b_arr, _ in test_scenarios:
            # Your implementation
            roots = np.full(len(a_arr), np.nan)
            iterations = np.full(len(a_arr), 100)
            converged = np.full(len(a_arr), False)
            results = (roots, iterations, converged)

            unconverged_idx = np.arange(len(a_arr))
            solver = BrentSolver()

            _try_back_up_bracket_vectorised(
                backup_solver=solver,
                func=func_jit,
                results=results,
                a=a_arr,
                b=b_arr,
                unconverged_idx=unconverged_idx,
                tol=1e-10,
                max_iter=100,
                func_params=None,
            )

            # SciPy implementation (brentq)
            scipy_roots = np.array(
                [optimize.brentq(func_scipy, a, b, xtol=1e-10) for a, b in zip(a_arr, b_arr)]
            )

            # Compare
            max_diff = np.max(np.abs(roots - scipy_roots))

            print(f"\n{name}:")
            print(f"  Your roots:  {roots}")
            print(f"  SciPy roots: {scipy_roots}")
            print(f"  Max diff:    {max_diff:.2e}")
            print(f"  Match:       {'✓' if max_diff < 1e-8 else '✗'}")

            assert np.all(converged), f"{name}: Not all converged"
            assert np.allclose(roots, scipy_roots, atol=1e-8), f"{name}: Precision mismatch"

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
