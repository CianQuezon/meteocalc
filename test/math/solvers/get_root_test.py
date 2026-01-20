"""
Comprehensive tests for RootSolvers.get_root method.

Tests cover:
- All three solver types (Newton, Brent, Bisection)
- Scalar and vectorized inputs
- Backup solver chain functionality
- Hybrid solver behavior (both open and bracket modes)
- Error handling and edge cases
- Accuracy validation against SciPy
- Performance characteristics
- Parameter passing (1D and 2D func_params)
- Convergence tracking

Author: Comprehensive Test Suite
"""

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from scipy.optimize import brentq, newton, bisect

from meteorological_equations.math.solvers.core import RootSolvers
from meteorological_equations.math.solvers._enums import SolverName


# ============================================================================
# Test Functions
# ============================================================================

@njit
def simple_quadratic(x):
    """x^2 - 4 = 0, roots at ±2"""
    return x**2 - 4


@njit
def simple_quadratic_prime(x):
    """Derivative: 2x"""
    return 2 * x


@njit
def cubic(x):
    """x^3 - 8 = 0, root at 2"""
    return x**3 - 8


@njit
def cubic_prime(x):
    """Derivative: 3x^2"""
    return 3 * x**2


@njit
def transcendental(x):
    """sin(x) - 0.5 = 0, multiple roots"""
    return np.sin(x) - 0.5


@njit
def transcendental_prime(x):
    """Derivative: cos(x)"""
    return np.cos(x)


@njit
def parametric_quadratic(x, a, b):
    """a*x^2 + b = 0"""
    return a * x**2 + b


@njit
def parametric_quadratic_prime(x, a, b):
    """Derivative: 2*a*x"""
    return 2 * a * x


@njit
def exponential_func(x):
    """e^x - 5 = 0, root at ln(5) ≈ 1.609"""
    return np.exp(x) - 5


@njit
def exponential_prime(x):
    """Derivative: e^x"""
    return np.exp(x)


@njit
def difficult_func(x):
    """x^3 - 2*x - 5 = 0, root ≈ 2.094551"""
    return x**3 - 2*x - 5


@njit
def difficult_prime(x):
    """Derivative: 3*x^2 - 2"""
    return 3*x**2 - 2


# ============================================================================
# Scalar Tests - Brent Solver
# ============================================================================

class TestGetRootScalarBrent:
    """Test scalar inputs with Brent solver."""
    
    def test_simple_quadratic_brent(self):
        """Test simple quadratic with Brent (positive root)."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='brent',
            use_backup=False
        )
        
        # Compare with SciPy
        scipy_root = brentq(simple_quadratic.py_func, 0.0, 5.0)
        
        assert conv is True, "Should converge"
        assert_allclose(root, scipy_root, rtol=1e-10, atol=1e-10)
        assert_allclose(root, 2.0, rtol=1e-6)
        assert iters > 0, "Should take some iterations"
        assert iters < 50, "Should converge quickly"
    
    def test_simple_quadratic_brent_negative_root(self):
        """Test simple quadratic with Brent (negative root)."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=-5.0,
            b=0.0,
            main_solver='brent',
            use_backup=False
        )
        
        scipy_root = brentq(simple_quadratic.py_func, -5.0, 0.0)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-10)
        assert_allclose(root, -2.0, rtol=1e-6)
    
    def test_cubic_brent(self):
        """Test cubic function with Brent."""
        root, iters, conv = RootSolvers.get_root(
            func=cubic,
            a=0.0,
            b=5.0,
            main_solver='brent',
            use_backup=False
        )
        
        scipy_root = brentq(cubic.py_func, 0.0, 5.0)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-9)
        assert_allclose(root, 2.0, rtol=1e-9)
    
    def test_transcendental_brent(self):
        """Test transcendental function with Brent."""
        root, iters, conv = RootSolvers.get_root(
            func=transcendental,
            a=0.0,
            b=1.0,
            main_solver='brent',
            tol=1e-10,
            use_backup=False
        )
        
        scipy_root = brentq(transcendental.py_func, 0.0, 1.0, xtol=1e-10)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-9)
        assert_allclose(root, np.pi/6, rtol=1e-6)  # sin(π/6) = 0.5
    
    def test_exponential_brent(self):
        """Test exponential function with Brent."""
        root, iters, conv = RootSolvers.get_root(
            func=exponential_func,
            a=0.0,
            b=3.0,
            main_solver='brent',
            use_backup=False
        )
        
        scipy_root = brentq(exponential_func.py_func, 0.0, 3.0)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-10)
        assert_allclose(root, np.log(5), rtol=1e-6)
    
    def test_tight_tolerance_brent(self):
        """Test Brent with very tight tolerance."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='brent',
            tol=1e-12,
            use_backup=False
        )
        
        assert conv is True
        assert_allclose(root, 2.0, rtol=1e-12, atol=1e-12)
        assert abs(simple_quadratic.py_func(root)) < 1e-12


# ============================================================================
# Scalar Tests - Newton Solver
# ============================================================================

class TestGetRootScalarNewton:
    """Test scalar inputs with Newton-Raphson solver."""
    
    def test_simple_quadratic_newton(self):
        """Test simple quadratic with Newton."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            func_prime=simple_quadratic_prime,
            x0=1.5,
            main_solver='newton',
            use_backup=False
        )
        
        # Compare with SciPy
        scipy_root = newton(simple_quadratic.py_func, 1.5, fprime=simple_quadratic_prime.py_func)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-10)
        assert_allclose(root, 2.0, rtol=1e-6)
        assert iters < 10, "Newton should converge very quickly"
    
    def test_cubic_newton(self):
        """Test cubic function with Newton."""
        root, iters, conv = RootSolvers.get_root(
            func=cubic,
            func_prime=cubic_prime,
            x0=2.5,
            main_solver='newton',
            use_backup=False
        )
        
        scipy_root = newton(cubic.py_func, 2.5, fprime=cubic_prime.py_func)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-10)
        assert_allclose(root, 2.0, rtol=1e-6)
    
    def test_transcendental_newton(self):
        """Test transcendental function with Newton."""
        root, iters, conv = RootSolvers.get_root(
            func=transcendental,
            func_prime=transcendental_prime,
            x0=0.5,
            main_solver='newton',
            tol=1e-10,
            use_backup=False
        )
        
        scipy_root = newton(transcendental.py_func, 0.5, fprime=transcendental_prime.py_func, tol=1e-10)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-9)
    
    def test_exponential_newton(self):
        """Test exponential function with Newton."""
        root, iters, conv = RootSolvers.get_root(
            func=exponential_func,
            func_prime=exponential_prime,
            x0=1.0,
            main_solver='newton',
            use_backup=False
        )
        
        scipy_root = newton(exponential_func.py_func, 1.0, fprime=exponential_prime.py_func)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-10)
        assert_allclose(root, np.log(5), rtol=1e-6)
    
    def test_difficult_newton(self):
        """Test difficult polynomial with Newton."""
        root, iters, conv = RootSolvers.get_root(
            func=difficult_func,
            func_prime=difficult_prime,
            x0=2.0,
            main_solver='newton',
            tol=1e-10,
            use_backup=False
        )
        
        scipy_root = newton(difficult_func.py_func, 2.0, fprime=difficult_prime.py_func, tol=1e-10)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-9)
        assert_allclose(root, 2.094551481542327, rtol=1e-6)


# ============================================================================
# Scalar Tests - Bisection Solver
# ============================================================================

class TestGetRootScalarBisection:
    """Test scalar inputs with Bisection solver."""
    
    def test_simple_quadratic_bisection(self):
        """Test simple quadratic with Bisection."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='bisection',
            use_backup=False
        )
        
        # Compare with SciPy
        scipy_root = bisect(simple_quadratic.py_func, 0.0, 5.0)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-6)
        assert_allclose(root, 2.0, rtol=1e-6)
        assert iters > 10, "Bisection takes more iterations"
    
    def test_cubic_bisection(self):
        """Test cubic function with Bisection."""
        root, iters, conv = RootSolvers.get_root(
            func=cubic,
            a=0.0,
            b=5.0,
            main_solver='bisection',
            use_backup=False
        )
        
        scipy_root = bisect(cubic.py_func, 0.0, 5.0)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-6)
        assert_allclose(root, 2.0, rtol=1e-6)
    
    def test_transcendental_bisection(self):
        """Test transcendental function with Bisection."""
        root, iters, conv = RootSolvers.get_root(
            func=transcendental,
            a=0.0,
            b=1.0,
            main_solver='bisection',
            tol=1e-10,
            use_backup=False
        )
        
        scipy_root = bisect(transcendental.py_func, 0.0, 1.0, xtol=1e-10)
        
        assert conv is True
        assert_allclose(root, scipy_root, rtol=1e-9)


# ============================================================================
# Vectorized Tests
# ============================================================================

class TestGetRootVectorized:
    """Test vectorized (array) inputs."""
    
    def test_vectorized_brent_same_function(self):
        """Test vectorized Brent with same function, different brackets."""
        n = 5
        a = np.array([0.0, 0.5, 1.0, 1.5, 1.8])
        b = np.array([5.0, 4.5, 4.0, 3.5, 3.0])
        
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=a,
            b=b,
            main_solver='brent',
            use_backup=False
        )
        
        # All should converge to 2.0
        assert roots.shape == (n,)
        assert np.all(conv), "All should converge"
        assert_allclose(roots, 2.0, rtol=1e-6)
        
        # Compare with SciPy for each
        for i in range(n):
            scipy_root = brentq(simple_quadratic.py_func, a[i], b[i])
            assert_allclose(roots[i], scipy_root, rtol=1e-10)
    
    def test_vectorized_newton_same_function(self):
        """Test vectorized Newton with same function, different initial guesses."""
        n = 5
        x0 = np.array([1.0, 1.5, 2.5, 3.0, 1.2])
        
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            func_prime=simple_quadratic_prime,
            x0=x0,
            main_solver='newton',
            use_backup=False
        )
        
        assert roots.shape == (n,)
        assert np.all(conv), "All should converge"
        assert_allclose(roots, 2.0, rtol=1e-6)
    
    def test_vectorized_bisection_same_function(self):
        """Test vectorized Bisection with same function."""
        n = 10
        a = np.linspace(0.0, 1.5, n)
        b = np.linspace(3.0, 5.0, n)
        
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=a,
            b=b,
            main_solver='bisection',
            use_backup=False
        )
        
        assert roots.shape == (n,)
        assert np.all(conv)
        assert_allclose(roots, 2.0, rtol=1e-6)
    
    def test_vectorized_large_array(self):
        """Test vectorized with large array (performance check)."""
        n = 1000
        a = np.full(n, 0.0)
        b = np.full(n, 5.0)
        
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=a,
            b=b,
            main_solver='brent',
            use_backup=False
        )
        
        assert roots.shape == (n,)
        assert np.all(conv), f"{np.sum(~conv)} out of {n} failed"
        assert_allclose(roots, 2.0, rtol=1e-6)
    
    def test_vectorized_parametric_1d_params(self):
        """Test vectorized with 1D func_params (shared parameters)."""
        n = 5
        a = np.zeros(n)
        b = np.full(n, 5.0)
        func_params = np.array([1.0, -4.0])  # a=1, b=-4 for all
        
        roots, iters, conv = RootSolvers.get_root(
            func=parametric_quadratic,
            a=a,
            b=b,
            func_params=func_params,
            main_solver='brent',
            use_backup=False
        )
        
        assert np.all(conv)
        assert_allclose(roots, 2.0, rtol=1e-6)
    
    def test_vectorized_parametric_2d_params(self):
        """Test vectorized with 2D func_params (different parameters per element)."""
        n = 3
        
        # Different (a, b) pairs, all solving to root at 2.0
        func_params = np.array([
            [1.0, -4.0],   # x^2 - 4 = 0
            [2.0, -8.0],   # 2x^2 - 8 = 0
            [0.5, -2.0],   # 0.5x^2 - 2 = 0
        ])
        
        a = np.zeros(n)
        b = np.full(n, 5.0)
        
        roots, iters, conv = RootSolvers.get_root(
            func=parametric_quadratic,
            a=a,
            b=b,
            func_params=func_params,
            main_solver='brent',
            use_backup=False
        )
        
        assert np.all(conv), f"Convergence: {conv}"
        assert_allclose(roots, 2.0, rtol=1e-6)


# ============================================================================
# Backup Solver Chain Tests
# ============================================================================

class TestGetRootBackupChain:
    """Test automatic backup solver functionality."""
    
    def test_backup_chain_all_converge_primary(self):
        """Test backup chain when primary solver succeeds (no backup needed)."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='brent',
            use_backup=True,
            backup_solvers=['bisection']
        )
        
        assert conv is True
        assert_allclose(root, 2.0, rtol=1e-6)
    
    def test_backup_chain_newton_to_brent_scalar(self):
        """Test Newton fails, Brent succeeds (scalar)."""
        # Newton with bad initial guess, far from root
        # Provide brackets for Brent backup
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            x0=100.0,  # Very far from root
            a=0.0,
            b=5.0,
            func_prime=simple_quadratic_prime,
            main_solver='newton',
            use_backup=True,
            backup_solvers=['brent'],
            max_iter=5  # Low max_iter to force Newton to fail
        )
        
        # Should still converge via Brent
        assert conv is True
        assert_allclose(root, 2.0, rtol=1e-6)
    
    def test_backup_chain_vectorized_partial_convergence(self):
        """Test vectorized with some elements needing backup."""
        n = 5
        
        # Mix of good and challenging initial guesses for Newton
        x0 = np.array([1.5, 100.0, 2.5, 200.0, 1.8])
        a = np.zeros(n)
        b = np.full(n, 5.0)
        
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            func_prime=simple_quadratic_prime,
            x0=x0,
            a=a,
            b=b,
            main_solver='newton',
            use_backup=True,
            backup_solvers=['brent', 'bisection'],
            max_iter=10
        )
        
        # All should converge (some via backup)
        assert np.all(conv), f"Not all converged: {conv}"
        assert_allclose(roots, 2.0, rtol=1e-5)
    
    def test_backup_chain_custom_order(self):
        """Test custom backup solver order."""
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='bisection',
            use_backup=True,
            backup_solvers=['brent'],  # Won't be needed
        )
        
        assert conv is True
        assert_allclose(roots, 2.0, rtol=1e-6)
    
    def test_backup_disabled(self):
        """Test with use_backup=False."""
        # Newton with impossible initial guess
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            x0=100.0,
            func_prime=simple_quadratic_prime,
            main_solver='newton',
            use_backup=False,
            max_iter=3
        )
        
        # Should return unconverged result
        assert conv is False or np.isnan(root) or not np.isclose(root, 2.0, rtol=1e-3)


# ============================================================================
# Hybrid Solver Tests (Brent with both x0 and brackets)
# ============================================================================

class TestGetRootHybridBrent:
    """Test Brent hybrid solver with both open and bracket interfaces."""
    
    def test_hybrid_open_succeeds_scalar(self):
        """Test hybrid Brent where open interface (x0) succeeds."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            x0=1.5,  # Good initial guess
            a=0.0,   # Also provide brackets
            b=5.0,
            main_solver='brent',
            use_backup=False
        )
        
        assert conv is True
        assert_allclose(root, 2.0, rtol=1e-6)
    
    def test_hybrid_fallback_to_bracket_scalar(self):
        """Test hybrid Brent falls back to bracket when open fails."""
        # Force open to fail with very low max_iter, bracket should work
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            x0=100.0,  # Bad initial guess
            a=0.0,
            b=5.0,
            main_solver='brent',
            use_backup=False,
            max_iter=100  # Enough for bracket
        )
        
        assert conv is True
        assert_allclose(root, 2.0, rtol=1e-6)
    
    def test_hybrid_vectorized_mixed_convergence(self):
        """Test hybrid with vectorized inputs where some use open, some use bracket."""
        n = 4
        x0 = np.array([1.5, 100.0, 2.5, 200.0])  # Mix of good and bad guesses
        a = np.zeros(n)
        b = np.full(n, 5.0)
        
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            x0=x0,
            a=a,
            b=b,
            main_solver='brent',
            use_backup=False
        )
        
        # All should converge (some via open, some via bracket)
        assert np.all(conv)
        assert_allclose(roots, 2.0, rtol=1e-6)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestGetRootErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_bracket_for_bisection(self):
        """Test that missing brackets for bisection creates substitute results."""
        with pytest.warns(UserWarning, match="Bracket solver"):
            root, iters, conv = RootSolvers.get_root(
                func=simple_quadratic,
                # Missing a and b
                main_solver='bisection',
                use_backup=False
            )
        
        # Should return unconverged
        assert conv is False
        assert np.isnan(root)
    
    def test_missing_x0_for_newton(self):
        """Test that missing x0 for Newton creates substitute results."""
        with pytest.warns(UserWarning, match="Open solver"):
            root, iters, conv = RootSolvers.get_root(
                func=simple_quadratic,
                func_prime=simple_quadratic_prime,
                # Missing x0
                main_solver='newton',
                use_backup=False
            )
        
        assert conv is False
        assert np.isnan(root)
    
    def test_all_inputs_none_raises_error(self):
        """Test that completely missing inputs eventually raises error."""
        with pytest.raises(ValueError, match="Cannot determine problem size"):
            RootSolvers.get_root(
                func=simple_quadratic,
                # No x0, a, or b
                main_solver='newton',
                use_backup=False
            )
    
    def test_invalid_bracket_no_sign_change(self):
        """Test invalid bracket (no sign change) is handled."""
        # Bracket [3, 5] has no sign change for x^2 - 4
        with pytest.warns(UserWarning):
            root, iters, conv = RootSolvers.get_root(
                func=simple_quadratic,
                a=3.0,
                b=5.0,
                main_solver='brent',
                use_backup=False,
                max_iter=10
            )
        
        # Should fail or warn
        assert not conv or np.isnan(root)


# ============================================================================
# Tolerance and Iteration Tests
# ============================================================================

class TestGetRootToleranceAndIterations:
    """Test tolerance and iteration limits."""
    
    def test_tight_tolerance(self):
        """Test with very tight tolerance."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='brent',
            tol=1e-14,
            use_backup=False
        )
        
        assert conv is True
        assert abs(simple_quadratic.py_func(root)) < 1e-14
        assert_allclose(root, 2.0, rtol=1e-14, atol=1e-14)
    
    def test_loose_tolerance(self):
        """Test with loose tolerance."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='brent',
            tol=1e-2,
            use_backup=False
        )
        
        assert conv is True
        assert_allclose(root, 2.0, rtol=1e-2)
        assert iters < 20, "Should converge quickly with loose tolerance"
    
    def test_low_max_iter_forces_failure(self):
        """Test that low max_iter can cause failure."""
        root, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='bisection',
            max_iter=3,  # Too few
            use_backup=False
        )
        
        # Might not converge with only 3 iterations
        if not conv:
            assert iters == 3
    
    def test_iteration_count_tracking(self):
        """Test that iteration counts are tracked correctly."""
        n = 3
        a = np.zeros(n)
        b = np.full(n, 5.0)
        
        roots, iters, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=a,
            b=b,
            main_solver='brent',
            use_backup=False
        )
        
        assert iters.shape == (n,)
        assert np.all(iters > 0), "Should have some iterations"
        assert np.all(iters < 50), "Should converge reasonably fast"


# ============================================================================
# Comparison with SciPy (Accuracy Validation)
# ============================================================================

class TestGetRootAccuracyValidation:
    """Comprehensive accuracy validation against SciPy."""
    
    @pytest.mark.parametrize("solver,scipy_func", [
        ('brent', brentq),
        ('bisection', bisect),
    ])
    def test_bracket_solvers_match_scipy(self, solver, scipy_func):
        """Test bracket solvers match SciPy results."""
        test_functions = [
            (simple_quadratic, 0.0, 5.0, 2.0),
            (cubic, 0.0, 5.0, 2.0),
            (exponential_func, 0.0, 3.0, np.log(5)),
        ]
        
        for func, a, b, expected in test_functions:
            root, _, conv = RootSolvers.get_root(
                func=func,
                a=a,
                b=b,
                main_solver=solver,
                tol=1e-10,
                use_backup=False
            )
            
            scipy_root = scipy_func(func.py_func, a, b, xtol=1e-10)
            
            assert conv, f"Failed to converge for {func}"
            assert_allclose(root, scipy_root, rtol=1e-10, atol=1e-10,
                          err_msg=f"Mismatch for {solver} on {func}")
            assert_allclose(root, expected, rtol=1e-6)
    
    def test_newton_matches_scipy(self):
        """Test Newton matches SciPy results."""
        test_functions = [
            (simple_quadratic, simple_quadratic_prime, 1.5, 2.0),
            (cubic, cubic_prime, 2.5, 2.0),
            (exponential_func, exponential_prime, 1.0, np.log(5)),
        ]
        
        for func, fprime, x0, expected in test_functions:
            root, _, conv = RootSolvers.get_root(
                func=func,
                func_prime=fprime,
                x0=x0,
                main_solver='newton',
                tol=1e-10,
                use_backup=False
            )
            
            scipy_root = newton(func.py_func, x0, fprime=fprime.py_func, tol=1e-10)
            
            assert conv
            assert_allclose(root, scipy_root, rtol=1e-10, atol=1e-10)
            assert_allclose(root, expected, rtol=1e-6)
    
    def test_vectorized_matches_scipy_element_wise(self):
        """Test vectorized results match SciPy element-wise."""
        n = 10
        a = np.linspace(0.0, 1.5, n)
        b = np.linspace(3.0, 5.0, n)
        
        roots, _, conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=a,
            b=b,
            main_solver='brent',
            tol=1e-10,
            use_backup=False
        )
        
        assert np.all(conv)
        
        # Compare each element with SciPy
        for i in range(n):
            scipy_root = brentq(simple_quadratic.py_func, a[i], b[i], xtol=1e-10)
            assert_allclose(roots[i], scipy_root, rtol=1e-10, atol=1e-10)


# ============================================================================
# Performance Characteristic Tests
# ============================================================================

class TestGetRootPerformance:
    """Test performance characteristics (not strict benchmarks)."""
    
    def test_newton_faster_than_bisection(self):
        """Test that Newton uses fewer iterations than Bisection."""
        newton_root, newton_iters, newton_conv = RootSolvers.get_root(
            func=simple_quadratic,
            func_prime=simple_quadratic_prime,
            x0=1.5,
            main_solver='newton',
            use_backup=False
        )
        
        bisect_root, bisect_iters, bisect_conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='bisection',
            use_backup=False
        )
        
        assert newton_conv and bisect_conv
        assert newton_iters < bisect_iters, "Newton should use fewer iterations"
        assert_allclose(newton_root, bisect_root, rtol=1e-6)
    
    def test_brent_faster_than_bisection(self):
        """Test that Brent uses fewer iterations than Bisection."""
        brent_root, brent_iters, brent_conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='brent',
            use_backup=False
        )
        
        bisect_root, bisect_iters, bisect_conv = RootSolvers.get_root(
            func=simple_quadratic,
            a=0.0,
            b=5.0,
            main_solver='bisection',
            use_backup=False
        )
        
        assert brent_conv and bisect_conv
        assert brent_iters < bisect_iters, "Brent should use fewer iterations"
        assert_allclose(brent_root, bisect_root, rtol=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================

class TestGetRootIntegration:
    """Integration tests simulating real usage."""
    
    def test_meteorological_dewpoint_calculation(self):
        """Simulate meteorological dew point calculation."""
        @njit
        def saturation_vp_diff(T, e):
            """Difference between saturation VP and actual VP."""
            es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
            return es - e
        
        # Calculate dew point for different vapor pressures
        vapor_pressures = np.array([500, 1000, 1500, 2000, 2500])
        n = len(vapor_pressures)
        
        # Create func_params (2D array)
        func_params = vapor_pressures.reshape(-1, 1)
        
        # Solve for each
        a = np.full(n, 250.0)  # Lower bound (K)
        b = np.full(n, 310.0)  # Upper bound (K)
        
        dewpoints, iters, conv = RootSolvers.get_root(
            func=saturation_vp_diff,
            a=a,
            b=b,
            func_params=func_params,
            main_solver='brent',
            tol=1e-8
        )
        
        assert np.all(conv), "All dew points should converge"
        assert np.all(dewpoints > 250) and np.all(dewpoints < 310)
        assert np.all(np.diff(dewpoints) > 0), "Higher VP = higher dew point"
    
    def test_robust_solver_chain_complex_problem(self):
        """Test robust solver chain on a complex problem."""
        @njit
        def complex_equation(x):
            """Complex equation that might challenge some solvers."""
            return np.tan(x) + x - 3
        
        # This equation can be tricky
        root, iters, conv = RootSolvers.get_root(
            func=complex_equation,
            a=0.1,
            b=1.0,
            main_solver='brent',
            use_backup=True,
            backup_solvers=['bisection'],
            tol=1e-8
        )
        
        assert conv, "Should converge with robust chain"
        assert abs(complex_equation.py_func(root)) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_"])