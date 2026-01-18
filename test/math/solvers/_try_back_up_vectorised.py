"""
Comprehensive unit tests for _try_back_up_vectorised function.

Tests validate correctness against SciPy reference implementations,
numerical accuracy, edge case handling, and performance characteristics.

Author: Cian Quezon
"""

import pytest
import numpy as np
import warnings
from numba import njit
from typing import Tuple
from scipy.optimize import brentq, bisect, newton

from meteorological_equations.math.solvers._back_up_logic import (
    _try_back_up_vectorised,
    _try_back_up_bracket_vectorised,
    _try_back_up_open_vectorised,
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
def trigonometric_func(x: float) -> float:
    """Trigonometric: sin(x) = 0, roots at x=nπ."""
    return np.sin(x)


@njit
def trigonometric_prime(x: float) -> float:
    """Derivative of trigonometric."""
    return np.cos(x)


@njit
def parametric_func(x: float, a: float, b: float) -> float:
    """Parametric: a*x^2 + b = 0."""
    return a * x**2 + b


@njit
def parametric_prime(x: float, a: float, b: float) -> float:
    """Derivative of parametric."""
    return 2.0 * a * x


@njit
def difficult_func(x: float) -> float:
    """Difficult function: x^3 - 2*x - 5 = 0, root near 2.09."""
    return x**3 - 2.0*x - 5.0


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
# Test Class: Basic Functionality
# ============================================================================

class TestBasicFunctionality:
    """Test basic functionality and correctness."""
    
    def test_all_unconverged_cubic_brent(self, default_tolerance, max_iterations):
        """Test all unconverged elements with Brent solver on cubic function."""
        n = 5
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        results = (roots, iters, conv)
        
        # Brackets around root at x=2
        a = np.array([0.0, 0.5, 1.0, 1.5, 1.8])
        b = np.array([3.0, 3.0, 3.0, 2.5, 2.3])
        
        # Run backup solver
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=results,
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            backup_solvers=[SolverName.BRENT]
        )
        
        # All should converge
        assert np.all(conv), "All elements should converge"
        
        # All roots should be near 2.0
        assert np.allclose(roots, 2.0, atol=default_tolerance), \
            f"Roots should be near 2.0, got {roots}"
        
        # Verify against SciPy
        scipy_roots = np.array([brentq(cubic_func, a[i], b[i]) for i in range(n)])
        assert np.allclose(roots, scipy_roots, atol=1e-8), \
            "Results should match SciPy brentq"
    
    def test_partial_convergence_preservation(self, default_tolerance, max_iterations):
        """Test that already-converged elements are preserved."""
        # Setup with some converged, some not
        roots = np.array([2.0, np.nan, 1.5, np.nan, np.nan])
        iters = np.array([8, 100, 10, 100, 100])
        conv = np.array([True, False, True, False, False])
        results = (roots, iters, conv)
        
        # Original converged values
        original_root_0 = roots[0]
        original_root_2 = roots[2]
        
        a = np.array([0, 0, 0, 1, 1.5])
        b = np.array([3, 3, 3, 3, 3])
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=results,
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        # Check preservation
        assert roots[0] == original_root_0, "Converged element 0 should be preserved"
        assert roots[2] == original_root_2, "Converged element 2 should be preserved"
        
        # Check new convergence
        assert np.all(conv), "All elements should now be converged"
        assert np.allclose(roots[[1, 3, 4]], 2.0, atol=default_tolerance), \
            "Newly converged elements should be correct"
    
    def test_all_converged_early_return(self, default_tolerance, max_iterations):
        """Test early return when all elements already converged."""
        roots = np.array([2.0, 2.0, 2.0])
        iters = np.array([5, 6, 7])
        conv = np.array([True, True, True])
        results_original = (roots.copy(), iters.copy(), conv.copy())
        results = (roots, iters, conv)
        
        a = np.array([0, 0, 0])
        b = np.array([3, 3, 3])
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=results,
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        # Nothing should change
        assert np.array_equal(roots, results_original[0])
        assert np.array_equal(iters, results_original[1])
        assert np.array_equal(conv, results_original[2])


# ============================================================================
# Test Class: SciPy Validation
# ============================================================================

class TestSciPyValidation:
    """Validate results against SciPy reference implementations."""
    
    def test_brent_vs_scipy_brentq(self, strict_tolerance):
        """Compare Brent method results with SciPy's brentq."""
        n = 20
        np.random.seed(42)
        
        # Random brackets around x=2
        a = np.random.uniform(0.0, 1.9, n)
        b = np.random.uniform(2.1, 4.0, n)
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        # Our implementation
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=strict_tolerance,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        # SciPy reference
        scipy_roots = np.array([brentq(cubic_func, a[i], b[i], xtol=strict_tolerance) 
                                for i in range(n)])
        
        assert np.all(conv), "All should converge"
        assert np.allclose(roots, scipy_roots, atol=1e-9), \
            f"Max difference from SciPy: {np.max(np.abs(roots - scipy_roots))}"
    
    def test_bisection_vs_scipy_bisect(self, default_tolerance):
        """Compare bisection method results with SciPy's bisect."""
        n = 15
        
        # Brackets for quadratic (roots at ±2)
        a = np.linspace(0.1, 1.5, n)
        b = np.full(n, 3.0)
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        # Our implementation
        roots, iters, conv = _try_back_up_vectorised(
            func=quadratic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=100,
            backup_solvers=[SolverName.BISECTION]
        )
        
        # SciPy reference
        scipy_roots = np.array([bisect(quadratic_func, a[i], b[i], xtol=default_tolerance) 
                                for i in range(n)])
        
        assert np.all(conv), "All should converge"
        assert np.allclose(roots, scipy_roots, atol=1e-8)
    
    def test_newton_vs_scipy_newton(self, strict_tolerance):
        """Compare Newton-Raphson with SciPy's newton."""
        n = 10
        
        # Initial guesses near root at x=2
        x0 = np.linspace(1.5, 2.5, n)
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        # Our implementation
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=None,
            b=None,
            x0=x0,
            tol=strict_tolerance,
            max_iter=50,
            func_prime=cubic_prime,
            backup_solvers=[SolverName.NEWTON]
        )
        
        # SciPy reference
        scipy_roots = np.array([newton(cubic_func, x0[i], fprime=cubic_prime, tol=strict_tolerance) 
                                for i in range(n)])
        
        assert np.all(conv), "All should converge"
        assert np.allclose(roots, scipy_roots, atol=1e-9)


# ============================================================================
# Test Class: Different Functions
# ============================================================================

class TestDifferentFunctions:
    """Test with various mathematical functions."""
    
    def test_exponential_function(self, default_tolerance, max_iterations):
        """Test with exponential function: e^x - 2 = 0."""
        n = 8
        expected_root = np.log(2.0)  # ≈ 0.693
        
        a = np.full(n, -1.0)
        b = np.full(n, 2.0)
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=exponential_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        assert np.all(conv)
        assert np.allclose(roots, expected_root, atol=default_tolerance)
        
        # Verify with SciPy
        scipy_root = brentq(exponential_func, -1.0, 2.0)
        assert np.allclose(roots[0], scipy_root, atol=1e-10)
    
    def test_trigonometric_function(self, default_tolerance, max_iterations):
        """Test with trigonometric function: sin(x) = 0."""
        n = 5
        
        # Brackets around different roots (0, π, 2π, ...)
        a = np.array([-0.5, 2.5, 5.5, 8.5, 12.0])
        b = np.array([0.5, 3.5, 6.5, 9.5, 13.0])
        expected_roots = np.array([0.0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=trigonometric_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        assert np.all(conv)
        assert np.allclose(roots, expected_roots, atol=default_tolerance)
    
    def test_difficult_function(self, default_tolerance, max_iterations):
        """Test with more challenging function: x^3 - 2x - 5 = 0."""
        # Root near x ≈ 2.0946
        n = 6
        
        a = np.linspace(1.5, 2.0, n)
        b = np.full(n, 3.0)
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=difficult_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        assert np.all(conv)
        
        # All should find same root
        assert np.allclose(roots, roots[0], atol=1e-8)
        
        # Verify with SciPy
        scipy_root = brentq(difficult_func, 1.5, 3.0)
        assert np.allclose(roots[0], scipy_root, atol=1e-10)


# ============================================================================
# Test Class: Parametric Functions
# ============================================================================

class TestParametricFunctions:
    """Test with functions that have additional parameters."""
    
    def test_1d_shared_parameters(self, default_tolerance, max_iterations):
        """Test with 1D parameters shared across all elements."""
        n = 5
        
        # All solve a*x^2 + b = 0 with same (a, b)
        func_params = np.tile([1.0, -4.0], (n, 1))  # x^2 - 4 = 0, roots at ±2
        
        a = np.full(n, 0.0)
        b = np.full(n, 5.0)
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=parametric_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_params=func_params
        )
        
        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)
    
    def test_2d_per_element_parameters(self, default_tolerance, max_iterations):
        """Test with 2D parameters (different per element)."""
        n = 4
        
        # Different (a, b) for each element
        func_params = np.array([
            [1.0, -4.0],   # x^2 - 4 = 0, root at 2
            [2.0, -8.0],   # 2x^2 - 8 = 0, root at 2
            [0.5, -2.0],   # 0.5x^2 - 2 = 0, root at 2
            [3.0, -12.0],  # 3x^2 - 12 = 0, root at 2
        ])
        
        a = np.full(n, 0.0)
        b = np.full(n, 5.0)
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=parametric_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_params=func_params
        )
        
        assert np.all(conv)
        # All should converge to root at x=2
        assert np.allclose(roots, 2.0, atol=default_tolerance)
    
    def test_partial_convergence_with_parameters(self, default_tolerance, max_iterations):
        """Test parameter extraction with partial convergence."""
        roots = np.array([2.0, np.nan, np.nan, 1.5])
        iters = np.array([5, 100, 100, 8])
        conv = np.array([True, False, False, True])
        
        func_params = np.array([
            [1.0, -4.0],   # Element 0 (already converged)
            [2.0, -8.0],   # Element 1 (needs solving)
            [0.5, -2.0],   # Element 2 (needs solving)
            [3.0, -12.0],  # Element 3 (already converged)
        ])
        
        a = np.array([0, 0, 0, 0])
        b = np.array([5, 5, 5, 5])
        
        roots, iters, conv = _try_back_up_vectorised(
            func=parametric_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_params=func_params
        )
        
        assert np.all(conv)
        # Elements 0 and 3 preserved
        assert roots[0] == 2.0
        assert roots[3] == 1.5
        # Elements 1 and 2 solved
        assert np.allclose(roots[[1, 2]], 2.0, atol=default_tolerance)


# ============================================================================
# Test Class: Solver Chain
# ============================================================================

class TestSolverChain:
    """Test backup solver chain behavior."""
    
    def test_single_solver_chain(self, default_tolerance, max_iterations):
        """Test with single solver in chain."""
        n = 3
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        a = np.array([0, 0, 0])
        b = np.array([3, 3, 3])
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations,
            backup_solvers=[SolverName.BRENT]
        )
        
        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)
    
    def test_multiple_solver_fallback(self, default_tolerance, max_iterations):
        """Test fallback through multiple solvers."""
        n = 5
        
        # Use good guesses for Newton first
        x0 = np.array([1.5, 1.8, 2.0, 2.2, 2.5])
        a = np.array([0, 0, 0, 0, 0])
        b = np.array([3, 3, 3, 3, 3])
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        # Chain: Newton → Brent → Bisection
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=x0,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_prime=cubic_prime,
            backup_solvers=[
                SolverName.NEWTON,
                SolverName.BRENT,
                SolverName.BISECTION
            ]
        )
        
        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)
    
    def test_hybrid_solver_both_interfaces(self, default_tolerance, max_iterations):
        """Test hybrid solver trying both open and bracket interfaces."""
        n = 4
        
        # Provide both x0 and brackets
        x0 = np.array([1.5, 2.5, 1.8, 2.2])
        a = np.array([0, 0, 0, 0])
        b = np.array([3, 3, 3, 3])
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        # Brent is hybrid - will try open interface first if x0 provided
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=x0,
            tol=default_tolerance,
            max_iter=max_iterations,
            func_prime=cubic_prime,
            backup_solvers=[SolverName.BRENT]
        )
        
        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)


# ============================================================================
# Test Class: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_array(self, default_tolerance, max_iterations):
        """Test with empty arrays."""
        roots = np.array([])
        iters = np.array([])
        conv = np.array([], dtype=bool)
        
        a = np.array([])
        b = np.array([])
        
        # Should handle gracefully
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        assert len(roots) == 0
        assert len(conv) == 0
    
    def test_single_element(self, default_tolerance, max_iterations):
        """Test with single element (boundary between scalar and vector)."""
        roots = np.array([np.nan])
        iters = np.array([100])
        conv = np.array([False])
        
        a = np.array([0.0])
        b = np.array([3.0])
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        assert conv[0] == True
        assert np.isclose(roots[0], 2.0, atol=default_tolerance)
    
    def test_very_tight_bracket(self, strict_tolerance, max_iterations):
        """Test with very tight initial brackets."""
        n = 3
        
        # Very tight brackets around x=2
        a = np.array([1.99, 1.999, 1.9999])
        b = np.array([2.01, 2.001, 2.0001])
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=strict_tolerance,
            max_iter=max_iterations
        )
        
        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=strict_tolerance)
        # Should converge quickly with tight brackets
        assert np.all(iters < 20)
    
    def test_wide_bracket(self, default_tolerance, max_iterations):
        """Test with very wide brackets."""
        n = 3
        
        # Very wide brackets
        a = np.array([-100.0, -50.0, -10.0])
        b = np.array([100.0, 50.0, 10.0])
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        assert np.all(conv)
        assert np.allclose(roots, 2.0, atol=default_tolerance)
    
    def test_max_iterations_exceeded(self, max_iterations):
        """Test behavior when max iterations is too low."""
        n = 3
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        a = np.array([0, 0, 0])
        b = np.array([3, 3, 3])
        
        # Very strict tolerance with few iterations
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            roots, iters, conv = _try_back_up_vectorised(
                func=cubic_func,
                results=(roots, iters, conv),
                a=a,
                b=b,
                x0=None,
                tol=1e-15,  # Very strict
                max_iter=3,  # Very few iterations
                backup_solvers=[SolverName.BISECTION]
            )
            
            # Should warn about unconverged
            assert any("did not converge" in str(warning.message) for warning in w)
    
    def test_invalid_bracket_graceful_failure(self, default_tolerance, max_iterations):
        """Test graceful handling of invalid brackets."""
        n = 3
        
        # Invalid brackets (same sign at both ends)
        a = np.array([3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0])
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            roots, iters, conv = _try_back_up_vectorised(
                func=cubic_func,
                results=(roots, iters, conv),
                a=a,
                b=b,
                x0=None,
                tol=default_tolerance,
                max_iter=max_iterations
            )
            
            # Should fail gracefully with warnings
            assert not np.all(conv)
            assert len(w) > 0


# ============================================================================
# Test Class: Large Scale Performance
# ============================================================================

class TestLargeScale:
    """Test performance and correctness at scale."""
    
    def test_large_array_1000_elements(self, default_tolerance):
        """Test with 1000 elements."""
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
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=100
        )
        
        elapsed = time.time() - start
        
        # Convergence check
        convergence_rate = np.sum(conv) / n * 100
        assert convergence_rate >= 99.0, f"Convergence rate: {convergence_rate}%"
        
        # Accuracy check
        converged_roots = roots[conv]
        assert np.allclose(converged_roots, 2.0, atol=default_tolerance)
        
        # Performance check (should be fast)
        assert elapsed < 2.0, f"Took {elapsed:.3f}s, expected < 2.0s"
        
        print(f"\n✓ Solved {n} roots in {elapsed:.3f}s")
        print(f"✓ Convergence rate: {convergence_rate:.1f}%")
        print(f"✓ Average iterations: {iters[conv].mean():.1f}")
    
    def test_mixed_convergence_large_scale(self, default_tolerance):
        """Test large scale with pre-converged elements."""
        n = 500
        np.random.seed(123)
        
        # Half already converged
        roots = np.random.randn(n)
        iters = np.random.randint(5, 50, n)
        conv = np.random.rand(n) < 0.5  # ~50% converged
        
        # Set unconverged to NaN
        roots[~conv] = np.nan
        iters[~conv] = 100
        
        n_unconverged_initial = np.sum(~conv)
        
        # Brackets for unconverged
        a = np.random.uniform(0.0, 1.9, n)
        b = np.random.uniform(2.1, 4.0, n)
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=100
        )
        
        n_converged_final = np.sum(conv)
        
        # Most should converge
        assert n_converged_final >= n_unconverged_initial * 0.95 + n - n_unconverged_initial


# ============================================================================
# Test Class: Numerical Accuracy
# ============================================================================

class TestNumericalAccuracy:
    """Test numerical accuracy and precision."""
    
    def test_high_precision_convergence(self):
        """Test convergence to very high precision."""
        n = 5
        tol = 1e-12
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        a = np.array([0, 0, 0, 0, 0])
        b = np.array([3, 3, 3, 3, 3])
        
        roots, iters, conv = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots, iters, conv),
            a=a,
            b=b,
            x0=None,
            tol=tol,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        assert np.all(conv)
        # Check residuals are very small
        residuals = np.array([cubic_func(r) for r in roots])
        assert np.all(np.abs(residuals) < tol)
    
    def test_consistency_across_runs(self, default_tolerance, max_iterations):
        """Test that results are consistent across multiple runs."""
        n = 10
        
        a = np.linspace(0, 1.5, n)
        b = np.full(n, 3.0)
        
        # Run twice
        roots1 = np.full(n, np.nan)
        iters1 = np.full(n, 100)
        conv1 = np.full(n, False)
        
        roots1, _, conv1 = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots1, iters1, conv1),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        roots2 = np.full(n, np.nan)
        iters2 = np.full(n, 100)
        conv2 = np.full(n, False)
        
        roots2, _, conv2 = _try_back_up_vectorised(
            func=cubic_func,
            results=(roots2, iters2, conv2),
            a=a,
            b=b,
            x0=None,
            tol=default_tolerance,
            max_iter=max_iterations
        )
        
        # Results should be identical
        assert np.array_equal(conv1, conv2)
        assert np.allclose(roots1, roots2, atol=1e-14)


# ============================================================================
# Test Class: Warning and Error Handling
# ============================================================================

class TestWarningsAndErrors:
    """Test warning and error handling."""
    
    def test_missing_brackets_warning(self, default_tolerance, max_iterations):
        """Test warning when brackets are missing for bracket methods."""
        n = 3
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # No brackets provided, only x0
            x0 = np.array([1.5, 2.0, 2.5])
            
            roots, iters, conv = _try_back_up_vectorised(
                func=cubic_func,
                results=(roots, iters, conv),
                a=None,
                b=None,
                x0=x0,
                tol=default_tolerance,
                max_iter=max_iterations,
                func_prime=cubic_prime,
                backup_solvers=[SolverName.BISECTION]  # Needs brackets
            )
            
            # Should warn about missing brackets
            assert any("requires brackets" in str(warning.message) for warning in w)
    
    def test_missing_initial_guess_warning(self, default_tolerance, max_iterations):
        """Test warning when initial guess missing for open methods."""
        n = 3
        
        roots = np.full(n, np.nan)
        iters = np.full(n, 100)
        conv = np.full(n, False)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Only brackets, no x0
            a = np.array([0, 0, 0])
            b = np.array([3, 3, 3])
            
            roots, iters, conv = _try_back_up_vectorised(
                func=cubic_func,
                results=(roots, iters, conv),
                a=a,
                b=b,
                x0=None,
                tol=default_tolerance,
                max_iter=max_iterations,
                backup_solvers=[SolverName.NEWTON]  # Needs x0
            )
            
            # Should warn about missing x0
            assert any("requires initial guess" in str(warning.message) for warning in w)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])