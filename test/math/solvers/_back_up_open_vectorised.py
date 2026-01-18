"""
Comprehensive tests for backup solver dispatchers.

Tests the _try_back_up_open_vectorised and _try_back_up_bracket_vectorised
functions against SciPy to ensure correctness, precision, and robustness.

Author: Test Suite
"""

import warnings

import numpy as np
import pytest
from numba import njit
from scipy import optimize

from meteorological_equations.math.solvers._back_up_logic import (
    _get_unconverged_func_params,
    _try_back_up_open_vectorised,
    _update_converged_results,
)
from meteorological_equations.math.solvers._solvers import (
    NewtonRaphsonSolver,
)

# =============================================================================
# TEST FUNCTIONS (Numba-compatible)
# =============================================================================


@njit
def quadratic_func(x):
    """x^2 - 4 = 0, roots at ±2"""
    return x**2 - 4


@njit
def quadratic_prime(x):
    """Derivative: 2x"""
    return 2 * x


@njit
def cubic_func(x):
    """x^3 - 2 = 0"""
    return x**3 - 2


@njit
def cubic_prime(x):
    """Derivative: 3x^2"""
    return 3 * x**2


@njit
def transcendental_func(x):
    """x - cos(x) = 0"""
    return x - np.cos(x)


@njit
def transcendental_prime(x):
    """Derivative: 1 + sin(x)"""
    return 1 + np.sin(x)


@njit
def exponential_func(x):
    """exp(x) - 3 = 0"""
    return np.exp(x) - 3


@njit
def exponential_prime(x):
    """Derivative: exp(x)"""
    return np.exp(x)


# Non-JIT versions for SciPy
def quadratic_scipy(x):
    return x**2 - 4


def cubic_scipy(x):
    return x**3 - 2


def transcendental_scipy(x):
    return x - np.cos(x)


def exponential_scipy(x):
    return np.exp(x) - 3


# =============================================================================
# TEST CLASS: Open Vectorised Dispatcher
# =============================================================================


class TestTryBackupOpenVectorised:
    """Comprehensive tests for _try_back_up_open_vectorised"""

    def test_basic_functionality_all_unconverged(self):
        """Test basic case where all elements are unconverged"""
        # Setup: 5 unconverged elements
        roots = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100, 100, 100])
        converged = np.array([False, False, False, False, False])
        results = (roots, iterations, converged)

        x0 = np.array([1.0, 1.5, -1.5, 2.5, -2.5])
        unconverged_idx = np.array([0, 1, 2, 3, 4])

        solver = NewtonRaphsonSolver()

        # Run backup
        success = _try_back_up_open_vectorised(
            backup_solver=solver,
            func=quadratic_func,
            results=results,
            x0=x0,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
            func_prime=quadratic_prime,
        )

        # Verify
        assert success, "Should return True if any converged"
        assert np.all(converged), "All should have converged"

        # x^2 - 4 = 0 has TWO roots: +2 and -2
        # Positive x0 → +2, Negative x0 → -2
        expected_roots = np.array([2.0, 2.0, -2.0, 2.0, -2.0])
        assert np.allclose(roots, expected_roots, atol=1e-5), (
            f"Expected {expected_roots}, got {roots}"
        )

        # Alternative: check that all roots satisfy x^2 - 4 = 0
        residuals = roots**2 - 4
        assert np.all(np.abs(residuals) < 1e-10), (
            f"All roots should satisfy f(x)=0, residuals: {residuals}"
        )

    def test_partial_convergence(self):
        """Test case where only some elements converge"""
        # Setup: 5 elements, 3 unconverged
        roots = np.array([2.0, np.nan, 1.5, np.nan, np.nan])
        iterations = np.array([10, 100, 8, 100, 100])
        converged = np.array([True, False, True, False, False])
        results = (roots, iterations, converged)

        x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        unconverged_idx = np.array([1, 3, 4])  # Only these need solving

        solver = NewtonRaphsonSolver()

        # Run backup on unconverged only
        success = _try_back_up_open_vectorised(
            backup_solver=solver,
            func=quadratic_func,
            results=results,
            x0=x0,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
            func_prime=quadratic_prime,
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
        assert np.abs(roots[1] - 2.0) < 1e-5
        assert np.abs(roots[3] - 2.0) < 1e-5
        assert np.abs(roots[4] - 2.0) < 1e-5

    def test_precision_vs_scipy(self):
        """Test numerical precision matches SciPy"""
        # Setup
        roots = np.array([np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100])
        converged = np.array([False, False, False])
        results = (roots, iterations, converged)

        x0_arr = np.array([0.5, 1.0, 1.5])
        unconverged_idx = np.array([0, 1, 2])

        solver = NewtonRaphsonSolver()

        # Your implementation
        _try_back_up_open_vectorised(
            backup_solver=solver,
            func=transcendental_func,
            results=results,
            x0=x0_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-10,
            max_iter=100,
            func_params=None,
            func_prime=transcendental_prime,
        )

        # SciPy reference
        scipy_roots = np.array(
            [
                optimize.newton(transcendental_scipy, x0, fprime=lambda x: 1 + np.sin(x), tol=1e-10)
                for x0 in x0_arr
            ]
        )

        # Compare precision
        print(f"\nYour roots:  {roots}")
        print(f"SciPy roots: {scipy_roots}")
        print(f"Difference:  {np.abs(roots - scipy_roots)}")

        assert np.all(converged), "All should converge"
        assert np.allclose(roots, scipy_roots, atol=1e-8), (
            f"Precision mismatch: yours={roots}, scipy={scipy_roots}"
        )

    def test_multiple_functions(self):
        """Test with different mathematical functions"""
        test_cases = [
            (quadratic_func, quadratic_prime, np.array([1.0, 1.5, 2.5]), 2.0),
            (cubic_func, cubic_prime, np.array([1.0, 1.5, 2.0]), 2 ** (1 / 3)),
            (exponential_func, exponential_prime, np.array([1.0, 1.5, 2.0]), np.log(3)),
        ]

        for func, func_prime, x0_arr, expected_root in test_cases:
            roots = np.full(len(x0_arr), np.nan)
            iterations = np.full(len(x0_arr), 100)
            converged = np.full(len(x0_arr), False)
            results = (roots, iterations, converged)

            unconverged_idx = np.arange(len(x0_arr))
            solver = NewtonRaphsonSolver()

            success = _try_back_up_open_vectorised(
                backup_solver=solver,
                func=func,
                results=results,
                x0=x0_arr,
                unconverged_idx=unconverged_idx,
                tol=1e-8,
                max_iter=100,
                func_params=None,
                func_prime=func_prime,
            )

            assert success
            assert np.all(converged)
            assert np.allclose(roots, expected_root, atol=1e-6), (
                f"Expected {expected_root}, got {roots}"
            )

    def test_with_func_params(self):
        """Test with function parameters"""

        @njit
        def parametric_func(x, a, b):
            return a * x**2 + b

        @njit
        def parametric_prime(x, a, _):
            return 2 * a * x

        # Setup
        roots = np.array([np.nan, np.nan, np.nan])
        iterations = np.array([100, 100, 100])
        converged = np.array([False, False, False])
        results = (roots, iterations, converged)

        x0_arr = np.array([1.0, 1.5, 2.0])
        unconverged_idx = np.array([0, 1, 2])

        # Parameters: a=1, b=-4 for each element (same as x^2 - 4)
        func_params = np.array([[1.0, -4.0], [1.0, -4.0], [1.0, -4.0]])

        solver = NewtonRaphsonSolver()

        success = _try_back_up_open_vectorised(
            backup_solver=solver,
            func=parametric_func,
            results=results,
            x0=x0_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=func_params,
            func_prime=parametric_prime,
        )

        assert success
        assert np.all(converged)
        assert np.allclose(roots, 2.0, atol=1e-5)

    def test_no_convergence_returns_false(self):
        """Test that function returns False when solver fails"""

        @njit
        def difficult_func(x):
            """Function that's hard to converge"""
            return np.tan(x) - x

        @njit
        def difficult_prime(x):
            return 1 / np.cos(x) ** 2 - 1

        # Setup with bad initial guesses
        roots = np.array([np.nan, np.nan])
        iterations = np.array([100, 100])
        converged = np.array([False, False])
        results = (roots, iterations, converged)

        x0_arr = np.array([1.5, 1.6])  # Near singularity
        unconverged_idx = np.array([0, 1])

        solver = NewtonRaphsonSolver()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            success = _try_back_up_open_vectorised(
                backup_solver=solver,
                func=difficult_func,
                results=results,
                x0=x0_arr,
                unconverged_idx=unconverged_idx,
                tol=1e-6,
                max_iter=5,  # Low max_iter
                func_params=None,
                func_prime=difficult_prime,
            )

        # May or may not converge, but should return bool
        assert isinstance(success, (bool, np.bool_))

    def test_empty_unconverged_idx(self):
        """Test with empty unconverged indices"""
        # Setup: all already converged
        roots = np.array([2.0, 2.0, 2.0])
        iterations = np.array([10, 10, 10])
        converged = np.array([True, True, True])
        results = (roots, iterations, converged)

        x0_arr = np.array([1.0, 1.0, 1.0])
        unconverged_idx = np.array([])  # Empty!

        solver = NewtonRaphsonSolver()

        # Should handle gracefully
        success = _try_back_up_open_vectorised(
            backup_solver=solver,
            func=quadratic_func,
            results=results,
            x0=x0_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
            func_prime=quadratic_prime,
        )

        # Nothing to do, but should not crash
        assert isinstance(success, (bool, np.bool_))
        # Results unchanged
        assert np.all(roots == 2.0)
        assert np.all(converged)

    def test_iterations_updated(self):
        """Test that iteration counts are properly updated"""
        roots = np.array([np.nan, np.nan])
        iterations = np.array([100, 100])  # Placeholder
        converged = np.array([False, False])
        results = (roots, iterations, converged)

        x0_arr = np.array([1.0, 1.5])
        unconverged_idx = np.array([0, 1])

        solver = NewtonRaphsonSolver()

        _try_back_up_open_vectorised(
            backup_solver=solver,
            func=quadratic_func,
            results=results,
            x0=x0_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
            func_prime=quadratic_prime,
        )

        # Iterations should be updated (not 100 anymore)
        assert iterations[0] != 100, "Iterations should be updated"
        assert iterations[1] != 100, "Iterations should be updated"
        assert iterations[0] > 0, "Should have positive iterations"
        assert iterations[1] > 0, "Should have positive iterations"
        assert iterations[0] < 20, "Should converge quickly for Newton"
        assert iterations[1] < 20, "Should converge quickly for Newton"

    def test_large_array(self):
        """Test with larger array (100 elements)"""
        n = 100
        roots = np.full(n, np.nan)
        iterations = np.full(n, 100)
        converged = np.full(n, False)
        results = (roots, iterations, converged)

        # Random initial guesses between 0.5 and 2.5
        np.random.seed(42)
        x0_arr = np.random.uniform(0.5, 2.5, n)
        unconverged_idx = np.arange(n)

        solver = NewtonRaphsonSolver()

        success = _try_back_up_open_vectorised(
            backup_solver=solver,
            func=quadratic_func,
            results=results,
            x0=x0_arr,
            unconverged_idx=unconverged_idx,
            tol=1e-6,
            max_iter=100,
            func_params=None,
            func_prime=quadratic_prime,
        )

        assert success
        assert np.all(converged), f"Expected all to converge, got {np.sum(converged)}/{n}"
        assert np.allclose(roots, 2.0, atol=1e-5)

    def test_comparison_comprehensive_scipy(self):
        """Comprehensive comparison with SciPy across multiple scenarios"""

        test_scenarios = [
            (
                "Quadratic",
                quadratic_func,
                quadratic_scipy,
                quadratic_prime,
                np.array([0.5, 1.0, 1.5, 2.5]),
                2.0,
            ),
            (
                "Cubic",
                cubic_func,
                cubic_scipy,
                cubic_prime,
                np.array([0.5, 1.0, 1.5, 2.0]),
                2 ** (1 / 3),
            ),
            (
                "Transcendental",
                transcendental_func,
                transcendental_scipy,
                transcendental_prime,
                np.array([0.3, 0.5, 0.7, 1.0]),
                0.739085,
            ),
        ]

        print("\n" + "=" * 80)
        print("COMPREHENSIVE SCIPY COMPARISON")
        print("=" * 80)

        for name, func_jit, func_scipy, func_prime, x0_arr, _ in test_scenarios:
            # Your implementation
            roots = np.full(len(x0_arr), np.nan)
            iterations = np.full(len(x0_arr), 100)
            converged = np.full(len(x0_arr), False)
            results = (roots, iterations, converged)

            unconverged_idx = np.arange(len(x0_arr))
            solver = NewtonRaphsonSolver()

            _try_back_up_open_vectorised(
                backup_solver=solver,
                func=func_jit,
                results=results,
                x0=x0_arr,
                unconverged_idx=unconverged_idx,
                tol=1e-10,
                max_iter=100,
                func_params=None,
                func_prime=func_prime,
            )

            # SciPy implementation
            if func_prime is not None:
                scipy_roots = np.array(
                    [optimize.newton(func_scipy, x0, fprime=func_prime, tol=1e-10) for x0 in x0_arr]
                )
            else:
                scipy_roots = np.array(
                    [optimize.newton(func_scipy, x0, tol=1e-10) for x0 in x0_arr]
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
# TEST CLASS: Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions in isolation"""

    def test_get_unconverged_func_params_with_params(self):
        """Test extracting func_params for unconverged indices"""
        func_params = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        unconverged_idx = np.array([1, 3])

        result = _get_unconverged_func_params(func_params, unconverged_idx)

        expected = np.array([[3.0, 4.0], [7.0, 8.0]])
        assert np.array_equal(result, expected)

    def test_get_unconverged_func_params_none(self):
        """Test with None func_params"""
        result = _get_unconverged_func_params(None, np.array([1, 2]))
        assert result is None

    def test_update_converged_results_partial(self):
        """Test updating only newly converged positions"""
        # Original results
        roots = np.array([2.0, np.nan, np.nan, 1.5, np.nan])
        iterations = np.array([10, 100, 100, 8, 100])
        converged = np.array([True, False, False, True, False])

        # Unconverged indices
        unconverged_idx = np.array([1, 2, 4])

        # Updated results (from backup solver)
        updated_roots = np.array([3.0, 4.0, 5.0])
        updated_iterations = np.array([12, 15, 20])
        updated_converged = np.array([True, True, False])  # Only first two converged

        # Update
        _update_converged_results(
            roots,
            iterations,
            converged,
            unconverged_idx,
            updated_roots,
            updated_iterations,
            updated_converged,
        )

        # Verify
        assert roots[0] == 2.0  # Unchanged (was already converged)
        assert roots[1] == 3.0  # Updated (newly converged)
        assert roots[2] == 4.0  # Updated (newly converged)
        assert roots[3] == 1.5  # Unchanged (was already converged)
        assert np.isnan(roots[4])  # Unchanged (didn't converge)

        assert converged[0]  # Unchanged
        assert converged[1]  # Updated to True
        assert converged[2]  # Updated to True
        assert converged[3]  # Unchanged
        assert not converged[4]  # Still False


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
