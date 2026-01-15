"""
Streamlined Unit Tests with Direct SciPy Comparison

Tests __try_back_up_scalar directly against SciPy without fixtures.
Uses Numba-compatible test functions.

Author: Test Suite
"""

import pytest
import numpy as np
import warnings
from scipy import optimize
from numba import njit

from meteorological_equations.math.solvers._enums import SolverName
from meteorological_equations.math.solvers._back_up_logic import _try_back_up_scalar

# =============================================================================
# NUMBA-COMPATIBLE TEST FUNCTIONS
# =============================================================================

@njit
def quadratic_func(x):
    """x^2 - 4 = 0, roots at ±2"""
    return x**2 - 4


@njit
def cubic_func(x):
    """x^3 - 2 = 0, root at 2^(1/3)"""
    return x**3 - 2


@njit
def transcendental_func(x):
    """x - cos(x) = 0"""
    return x - np.cos(x)


@njit
def exponential_func(x):
    """exp(x) - 3 = 0"""
    return np.exp(x) - 3


@njit
def polynomial_func(x):
    """x^3 - 6x^2 + 11x - 6 = 0, roots at 1, 2, 3"""
    return x**3 - 6*x**2 + 11*x - 6


@njit
def sine_func(x):
    """sin(x) = 0, roots at multiples of π"""
    return np.sin(x)


@njit
def lambert_w_func(x):
    """x * exp(x) - 1 = 0, Lambert W function"""
    return x * np.exp(x) - 1


@njit
def steep_func(x):
    """x^10 - 1 = 0, very steep near root"""
    return x**10 - 1


@njit
def flat_func(x):
    """(x - 1)^5 = 0, very flat near root"""
    return (x - 1)**5


# Python versions for SciPy (SciPy doesn't need @njit)
def quadratic_func_scipy(x):
    return x**2 - 4

def cubic_func_scipy(x):
    return x**3 - 2

def transcendental_func_scipy(x):
    return x - np.cos(x)

def exponential_func_scipy(x):
    return np.exp(x) - 3

def polynomial_func_scipy(x):
    return x**3 - 6*x**2 + 11*x - 6

def sine_func_scipy(x):
    return np.sin(x)

def lambert_w_func_scipy(x):
    return x * np.exp(x) - 1

def steep_func_scipy(x):
    return x**10 - 1

def flat_func_scipy(x):
    return (x - 1)**5


# =============================================================================
# DIRECT SCIPY COMPARISON TESTS
# =============================================================================

class TestDirectScipyComparison:
    """Test by comparing directly with SciPy (no fixtures)"""
    
    def test_quadratic_brent_vs_scipy(self):
        """Quadratic: x^2 - 4 = 0"""
        failed_results = (0.0, 100, False)
    def test_quadratic_brent_vs_scipy(self):
        """Quadratic: x^2 - 4 = 0"""
        failed_results = (0.0, 100, False)
        
        # Valid brackets: func(0)=-4 (neg), func(5)=21 (pos)
        a, b = 0.0, 5.0
        
        # Your implementation (uses @njit function)
        result = _try_back_up_scalar(
            func=quadratic_func,  # ✅ Numba-compatible
            results=failed_results,
            a=a,
            b=b,
            x0=1.0,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        # SciPy reference (uses regular Python function)
        scipy_root = optimize.brentq(quadratic_func_scipy, a, b, xtol=1e-6)
        
        # Compare
        print(f"\n✓ Quadratic: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True, f"Should converge, got: {result}"
        assert np.isclose(result[0], scipy_root, atol=1e-5), \
            f"Your: {result[0]:.10f}, SciPy: {scipy_root:.10f}"
    
    
    def test_cubic_brent_vs_scipy(self):
        """Cubic: x^3 - 2 = 0"""
        failed_results = (1.0, 100, False)
        a, b = 0.0, 2.0
        
        # Your implementation
        result = _try_back_up_scalar(
            func=cubic_func,
            results=failed_results,
            a=a,
            b=b,
            x0=1.0,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        # SciPy reference
        scipy_root = optimize.brentq(cubic_func_scipy, a, b, xtol=1e-6)
        
        print(f"✓ Cubic: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)
    
    
    def test_transcendental_brent_vs_scipy(self):
        """Transcendental: x - cos(x) = 0"""
        failed_results = (0.5, 100, False)
        a, b = 0.0, 1.5
        
        result = _try_back_up_scalar(
            func=transcendental_func,
            results=failed_results,
            a=a,
            b=b,
            x0=0.5,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        scipy_root = optimize.brentq(transcendental_func_scipy, a, b, xtol=1e-6)
        
        print(f"✓ Transcendental: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)
    
    
    def test_exponential_brent_vs_scipy(self):
        """Exponential: exp(x) - 3 = 0"""
        failed_results = (0.5, 100, False)
        a, b = 0.0, 2.0
        
        result = _try_back_up_scalar(
            func=exponential_func,
            results=failed_results,
            a=a,
            b=b,
            x0=1.0,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        scipy_root = optimize.brentq(exponential_func_scipy, a, b, xtol=1e-6)
        
        print(f"✓ Exponential: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)
    
    
    def test_polynomial_brent_vs_scipy(self):
        """Polynomial: x^3 - 6x^2 + 11x - 6 = 0 (root at 2)"""
        failed_results = (1.5, 100, False)
        a, b = 1.5, 2.5
        
        result = _try_back_up_scalar(
            func=polynomial_func,
            results=failed_results,
            a=a,
            b=b,
            x0=2.0,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        scipy_root = optimize.brentq(polynomial_func_scipy, a, b, xtol=1e-6)
        
        print(f"✓ Polynomial: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)
    
    
    def test_trigonometric_brent_vs_scipy(self):
        """Trigonometric: sin(x) = 0 (root at π)"""
        failed_results = (3.0, 100, False)
        a, b = 3.0, 3.3
        
        result = _try_back_up_scalar(
            func=sine_func,
            results=failed_results,
            a=a,
            b=b,
            x0=3.1,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        scipy_root = optimize.brentq(sine_func_scipy, a, b, xtol=1e-6)
        
        print(f"✓ Trigonometric: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)


# =============================================================================
# BISECTION VS SCIPY
# =============================================================================

class TestBisectionVsScipy:
    """Test bisection method directly against SciPy"""
    
    def test_quadratic_bisection_vs_scipy(self):
        """Compare bisection with scipy.optimize.bisect"""
        failed_results = (0.0, 100, False)
        a, b = 0.0, 5.0
        
        result = _try_back_up_scalar(
            func=quadratic_func,
            results=failed_results,
            a=a,
            b=b,
            x0=None,  # Force bisection
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BISECTION]
        )
        
        scipy_root = optimize.bisect(quadratic_func_scipy, a, b, xtol=1e-6)
        
        print(f"✓ Bisection: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)
    
    
    def test_cubic_bisection_vs_scipy(self):
        """Cubic with bisection"""
        failed_results = (1.0, 100, False)
        a, b = 0.0, 2.0
        
        result = _try_back_up_scalar(
            func=cubic_func,
            results=failed_results,
            a=a,
            b=b,
            x0=None,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BISECTION]
        )
        
        scipy_root = optimize.bisect(cubic_func_scipy, a, b, xtol=1e-6)
        
        print(f"✓ Bisection Cubic: Your={result[0]:.10f}, SciPy={scipy_root:.10f}")
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)


# =============================================================================
# PRECISION COMPARISON
# =============================================================================

class TestPrecisionVsScipy:
    """Test numerical precision matches SciPy"""
    
    @pytest.mark.parametrize("tol", [1e-6, 1e-8, 1e-10])
    def test_precision_at_various_tolerances(self, tol):
        """Test precision at different tolerance levels"""
        true_root = np.sqrt(2)  # Root of x^2 - 2
        failed_results = (1.0, 100, False)
        a, b = 0.0, 3.0
        
        # Function: x^2 - 2 = 0
        @njit
        def sqrt_func(x):
            return x**2 - 2
        
        def sqrt_func_scipy(x):
            return x**2 - 2
        
        # Your implementation
        result = _try_back_up_scalar(
            func=sqrt_func,
            results=failed_results,
            a=a,
            b=b,
            x0=1.0,
            tol=tol,
            max_iter=1000,
            backup_solvers=[SolverName.BRENT]
        )
        
        # SciPy implementation
        scipy_root = optimize.brentq(sqrt_func_scipy, a, b, xtol=tol)
        
        # Both should be close to true root
        your_error = abs(result[0] - true_root)
        scipy_error = abs(scipy_root - true_root)
        
        print(f"\nTolerance: {tol:.0e}")
        print(f"  True root:  {true_root:.15f}")
        print(f"  Your root:  {result[0]:.15f} (error: {your_error:.2e})")
        print(f"  SciPy root: {scipy_root:.15f} (error: {scipy_error:.2e})")
        
        # Your implementation should be comparable to SciPy
        assert result[2] is True
        assert your_error < tol * 100  # Within reasonable factor
        assert np.isclose(result[0], scipy_root, atol=tol * 10)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases with direct SciPy comparison"""
    
    def test_root_at_boundary(self):
        """Root exactly at bracket boundary"""
        @njit
        def boundary_func(x):
            return x - 2.0
        
        def boundary_func_scipy(x):
            return x - 2.0
        
        failed_results = (0.0, 100, False)
        a, b = 0.0, 2.0  # Root at b
        
        result = _try_back_up_scalar(
            func=boundary_func,
            results=failed_results,
            a=a,
            b=b,
            x0=1.0,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        scipy_root = optimize.brentq(boundary_func_scipy, a, b, xtol=1e-6)
        
        assert result[2] is True
        assert np.isclose(result[0], 2.0, atol=1e-6)
        assert np.isclose(result[0], scipy_root, atol=1e-6)
    
    
    def test_steep_function(self):
        """Very steep function: x^10 - 1 = 0"""
        failed_results = (0.5, 100, False)
        a, b = 0.5, 1.5
        
        result = _try_back_up_scalar(
            func=steep_func,
            results=failed_results,
            a=a,
            b=b,
            x0=1.0,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        scipy_root = optimize.brentq(steep_func_scipy, a, b, xtol=1e-6)
        
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)
    
    
    def test_flat_function(self):
        """Very flat function: (x-1)^5 = 0"""
        failed_results = (0.0, 100, False)
        a, b = 0.0, 2.0
        
        result = _try_back_up_scalar(
            func=flat_func,
            results=failed_results,
            a=a,
            b=b,
            x0=0.5,
            tol=1e-6,
            max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        scipy_root = optimize.brentq(flat_func_scipy, a, b, xtol=1e-6)
        
        assert result[2] is True
        assert np.isclose(result[0], scipy_root, atol=1e-5)


# =============================================================================
# COMPREHENSIVE VALIDATION REPORT
# =============================================================================

class TestComprehensiveReport:
    """Generate comprehensive comparison report"""
    
    def test_generate_comparison_report(self):
        """Compare across multiple functions and report"""
        
        test_cases = [
            ("Quadratic", quadratic_func, quadratic_func_scipy, (0.0, 5.0)),
            ("Cubic", cubic_func, cubic_func_scipy, (0.0, 2.0)),
            ("Transcendental", transcendental_func, transcendental_func_scipy, (0.0, 1.0)),
            ("Exponential", exponential_func, exponential_func_scipy, (0.0, 2.0)),
            ("Sine", sine_func, sine_func_scipy, (3.0, 3.3)),
            ("Lambert W", lambert_w_func, lambert_w_func_scipy, (0.0, 1.0)),
        ]
        
        print("\n" + "="*70)
        print("COMPREHENSIVE SCIPY COMPARISON REPORT")
        print("="*70)
        
        all_passed = True
        
        for name, func_numba, func_scipy, (a, b) in test_cases:
            failed_results = (0.0, 100, False)
            
            # Your implementation
            try:
                result = _try_back_up_scalar(
                    func=func_numba,
                    results=failed_results,
                    a=a, b=b, x0=(a+b)/2,
                    tol=1e-10, max_iter=100,
                    backup_solvers=[SolverName.BRENT]
                )
            except Exception as e:
                print(f"\n{name}: FAILED - {e}")
                all_passed = False
                continue
            
            # SciPy implementation
            try:
                scipy_root = optimize.brentq(func_scipy, a, b, xtol=1e-10)
            except Exception as e:
                print(f"\n{name}: SciPy FAILED - {e}")
                continue
            
            # Compare
            diff = abs(result[0] - scipy_root)
            match = "✓" if diff < 1e-8 else "✗"
            
            print(f"\n{name}:")
            print(f"  Your root:   {result[0]:.15f}")
            print(f"  SciPy root:  {scipy_root:.15f}")
            print(f"  Difference:  {diff:.2e}")
            print(f"  Match:       {match}")
            
            if result[2]:
                assert np.isclose(result[0], scipy_root, atol=1e-8), \
                    f"{name} mismatch"
            else:
                all_passed = False
        
        print("\n" + "="*70)
        if all_passed:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
        print("="*70)


# =============================================================================
# BASIC FUNCTIONALITY (kept minimal)
# =============================================================================

class TestBasicFunctionality:
    """Basic tests that don't need SciPy comparison"""
    
    def test_already_converged_returns_immediately(self):
        """Already converged should return immediately"""
        converged_results = (2.0, 5, True)
        
        result = _try_back_up_scalar(
            func=quadratic_func,
            results=converged_results,
            a=0.0, b=5.0, x0=1.0,
            tol=1e-6, max_iter=100,
            backup_solvers=[SolverName.BRENT]
        )
        
        assert result == converged_results
        assert result[2] is True
    
    
    def test_returns_original_if_all_fail(self):
        """Should return original if all backups fail"""
        @njit
        def no_roots_func(x):
            return x**2 + 1  # No real roots
        
        failed_results = (0.0, 100, False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _try_back_up_scalar(
                func=no_roots_func,
                results=failed_results,
                a=-10.0, b=10.0, x0=0.0,
                tol=1e-6, max_iter=100,
                backup_solvers=[SolverName.BISECTION]
            )
        
        assert result[2] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])