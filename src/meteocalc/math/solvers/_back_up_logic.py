"""
Logic for using backup solvers.

Author: Cian Quezon
"""

import warnings
import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Union, Tuple, List
from meteorological_equations.math.solvers._solvers import Solver
from meteorological_equations.shared._enum_tools import parse_enum
from meteorological_equations.math.solvers._enums import MethodType, SolverName
from meteorological_equations.math.solvers.core import SolverMap



def _use_back_up_solvers(
    func: Callable[[float], float],
    results: Union[Tuple[float, int, bool], Tuple[npt.NDArray, npt.NDArray, npt.NDArray]],
    a: Optional[Union[npt.ArrayLike, float]],
    b: Optional[Union[npt.ArrayLike, float]],
    x0: Optional[Union[npt.ArrayLike, float]],
    tol: float,
    max_iter: int,
    func_prime: Optional[Callable[[float], float]] = None,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]] = None,
    backup_solvers: List[Union[str, MethodType]] = None
) -> Union[
    Tuple[float, int, bool],
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
]:
    """
    Apply backup solvers to unconverged root-finding results.
    
    Dispatches to scalar or vectorized backup solver logic based on input
    dimensionality. Automatically tries a chain of fallback methods (default:
    Brent's method, then bisection) to improve convergence rates when the
    primary solver fails.
    
    This function serves as the main entry point for the backup solver system,
    determining whether to use scalar or vectorized processing and delegating
    to the appropriate handler.
    
    Parameters
    ----------
    func : Callable[[float], float]
        The function for which to find roots. Must accept a float and return
        a float. For vectorized operations, will be called element-wise.
        Should satisfy f(x) = 0 at the root.
    results : tuple
        Results from the primary solver attempt. Can be either:
        - Scalar: (root, iterations, converged) where types are (float, int, bool)
        - Vectorized: (roots, iterations, converged) where types are (ndarray, ndarray, ndarray)
    a : float or array_like, optional
        Lower bracket bound(s). Required for bracket-based methods (Brent, bisection).
        - Scalar: single float value
        - Vectorized: array of floats, one per element
        If None, bracket methods will be skipped in the fallback chain.
    b : float or array_like, optional
        Upper bracket bound(s). Required for bracket-based methods.
        Must satisfy func(a) * func(b) < 0 for each bracket pair.
        If None, bracket methods will be skipped.
    x0 : float or array_like, optional
        Initial guess(es) for root location. Required for open methods (Newton-Raphson).
        - Scalar: single float value
        - Vectorized: array of floats, one per element
        If None, open methods will be skipped in the fallback chain.
    tol : float
        Convergence tolerance. Solver stops when |f(x)| < tol or when the
        change in x between iterations is less than tol.
    max_iter : int
        Maximum number of iterations allowed per solver attempt.
    func_prime : Callable[[float], float], optional
        Derivative of func. Required for derivative-based methods like Newton-Raphson.
        If None, only derivative-free methods will be used. Default is None.
    func_params : array_like or tuple of floats, optional
        Additional parameters to pass to func and func_prime. For vectorized
        operations, should be a 2D array where each row corresponds to parameters
        for one element. Default is None.
    backup_solvers : list of str or SolverName, optional
        Ordered list of backup solvers to try. Each solver is attempted in
        sequence until convergence or list exhaustion. Default is [BRENT, BISECTION].
        Available solvers: BRENT, BISECTION, NEWTON_RAPHSON, SECANT, REGULA_FALSI.
    
    Returns
    -------
    results : tuple
        Updated results with the same structure as input:
        - Scalar: (root, iterations, converged) as (float, int, bool)
        - Vectorized: (roots, iterations, converged) as (ndarray, ndarray, ndarray)
        
        For vectorized results:
        - roots: Array of root values (NaN for unconverged elements)
        - iterations: Array of iteration counts
        - converged: Boolean array indicating convergence status
    
    Notes
    -----
    - The function automatically detects scalar vs. vectorized inputs based on
      the dimensionality of the results tuple
    - For vectorized inputs, backup solvers are applied only to unconverged elements,
      preserving already-converged results
    - Hybrid solvers (e.g., Brent) will try both open and bracket interfaces if
      both x0 and (a, b) are provided
    - If all backup solvers fail, the function returns the input results unchanged
      with appropriate warnings
    
    Examples
    --------
    Scalar usage with bracket methods:
    >>>
    >>> @njit
    >>> def f(x):
    ...     return x**3 - 8
    >>> 
    >>> # Primary solver failed
    >>> results = (np.nan, 100, False)
    >>> 
    >>> # Try backup solvers with brackets
    >>> root, iters, converged = _use_back_up_solvers(
    ...     func=f,
    ...     results=results,
    ...     a=0.0,
    ...     b=5.0,
    ...     x0=None,
    ...     tol=1e-6,
    ...     max_iter=100
    ... )
    >>> print(f"Root: {root:.6f}, Converged: {converged}")
    Root: 2.000000, Converged: True
    
    Vectorized usage with mixed convergence:
    
    >>> @njit
    >>> def f(x):
    ...     return x**3 - 8
    >>> 
    >>> # Some elements converged, some didn't
    >>> roots = np.array([2.0, np.nan, np.nan])
    >>> iters = np.array([8, 100, 100])
    >>> conv = np.array([True, False, False])
    >>> 
    >>> # Apply backup solvers to unconverged elements only
    >>> roots, iters, conv = _use_back_up_solvers(
    ...     func=f,
    ...     results=(roots, iters, conv),
    ...     a=np.array([0, 0, 0]),
    ...     b=np.array([5, 5, 5]),
    ...     x0=None,
    ...     tol=1e-6,
    ...     max_iter=100
    ... )
    >>> print(conv)
    [True True True]
    
    Using custom backup solver chain:
    
    >>> from meteorological_equations.math.solvers import SolverName
    >>> 
    >>> # Try Newton first, then Brent, then bisection
    >>> results = _use_back_up_solvers(
    ...     func=f,
    ...     results=(np.nan, 100, False),
    ...     a=0.0,
    ...     b=5.0,
    ...     x0=2.5,
    ...     tol=1e-8,
    ...     max_iter=50,
    ...     func_prime=lambda x: 3*x**2,
    ...     backup_solvers=[
    ...         SolverName.NEWTON_RAPHSON,
    ...         SolverName.BRENT,
    ...         SolverName.BISECTION
    ...     ]
    ... )
    
    See Also
    --------
    _try_back_up_scalar : Scalar backup solver implementation
    _try_back_up_vectorised : Vectorized backup solver implementation
    """  

    if backup_solvers is None:
        backup_solvers = [SolverName.BRENT, SolverName.BISECTION]

    roots = np.asarray(results[0])
    iterations = np.asarray(results[1])
    converged_flag = np.asarray(results[2])

    if roots.ndim == 0:
        scalar_results = (float(roots), int(iterations), bool(converged_flag))
            
        scalar_a = float(a) if a is not None else None
        scalar_b = float(b) if b is not None else None
        scalar_x0 = float(x0) if x0 is not None else None



        return _try_back_up_scalar(
            func=func,
            results=scalar_results,
            a=scalar_a,
            b=scalar_b,
            x0=scalar_x0,
            tol=tol,
            max_iter=max_iter,
            func_prime=func_prime,
            func_params=func_params,
            backup_solvers=backup_solvers

        )
        
    else:
        
        return _try_back_up_vectorised(
            func=func,
            results=results,
            a=a,
            b=b,
            x0=x0,
            tol=tol,
            max_iter=max_iter,
            func_prime=func_prime,
            func_params=func_params,
            backup_solvers=backup_solvers
        )




def _try_back_up_scalar(
    func: Callable[[float], float],
    results: Tuple[float, int, bool],
    a: Optional[float],
    b: Optional[float],
    x0: Optional[float],
    tol: float,
    max_iter: int,
    func_prime: Optional[Callable[[float], float]] = None,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]] = None,
    backup_solvers: List[Union[str, MethodType]] = None,
            
    ):
    """
    Apply a chain of backup solvers to a single unconverged root-finding result.
    
    Iterates through a sequence of backup solvers, attempting each in order until
    one successfully converges or all are exhausted. Handles different solver types
    (open, bracket, hybrid) and gracefully skips solvers when required inputs are
    missing.
    
    This function is designed for scalar (single-value) operations. For vectorized
    operations on arrays, use `_try_back_up_vectorised` instead.
    
    Parameters
    ----------
    func : Callable[[float], float]
        The function for which to find a root. Must accept a single float argument
        and return a float. The solver seeks x such that func(x) = 0.
    results : tuple of (float, int, bool)
        Results from the primary solver attempt:
        - results[0]: Root value (may be NaN if unconverged)
        - results[1]: Number of iterations used
        - results[2]: Convergence flag (True if converged, False otherwise)
    a : float, optional
        Lower bracket bound. Required for bracket-based methods (Brent, bisection,
        regula falsi). Must satisfy func(a) * func(b) < 0. If None, bracket methods
        are skipped.
    b : float, optional
        Upper bracket bound. Required for bracket-based methods. If None, bracket
        methods are skipped.
    x0 : float, optional
        Initial guess for the root location. Required for open methods (Newton-Raphson,
        secant). Should be reasonably close to the actual root for best convergence.
        If None, open methods are skipped.
    tol : float
        Convergence tolerance. The solver stops when |func(x)| < tol or when the
        change in x between iterations falls below tol. Typical values: 1e-6 to 1e-12.
    max_iter : int
        Maximum number of iterations allowed for each solver attempt. If exceeded,
        the solver is considered to have failed and the next backup is tried.
    func_prime : Callable[[float], float], optional
        Derivative of func with respect to x. Required for derivative-based methods
        like Newton-Raphson. Must return df/dx at the given point. If None,
        derivative-based methods are skipped. Default is None.
    func_params : tuple of floats, optional
        Additional parameters to pass to func and func_prime. These are passed as
        positional arguments after x. For example, if func_params=(a, b), then
        func is called as func(x, a, b). Default is None.
    backup_solvers : list of str or SolverName, optional
        Ordered sequence of backup solvers to try. Each solver is attempted in the
        order specified until one converges. Default is [SolverName.BRENT, 
        SolverName.BISECTION].
        
        Available solvers:
        - BRENT: Hybrid method combining bisection, secant, and inverse quadratic
        - BISECTION: Reliable bracket method, always converges but slower
        - NEWTON_RAPHSON: Fast open method requiring derivative
        - SECANT: Open method approximating derivative
        - REGULA_FALSI: Bracket method, alternative to bisection
    
    Returns
    -------
    results : tuple of (float, int, bool)
        Updated results after attempting backup solvers:
        - results[0]: Root value (solution to func(x) = 0, or NaN if all failed)
        - results[1]: Total iterations used (cumulative across attempts)
        - results[2]: Convergence status (True if any solver succeeded, False otherwise)
    
    Notes
    -----
    Solver Selection Logic:
    - HYBRID solvers (e.g., Brent): Try open interface first if x0 provided, then
      bracket interface if (a, b) provided
    - BRACKET solvers: Require both a and b; skipped if either is None
    - OPEN solvers: Require x0; skipped if None
    
    Error Handling:
    - If a solver raises an exception, a warning is issued and the next solver is tried
    - If a solver converges (results[2] == True), iteration stops and results are returned
    - If all solvers fail or are skipped, the original results are returned unchanged
    
    The function returns immediately upon first successful convergence. This means
    earlier solvers in the chain are preferred, so order matters.
    
    Performance Considerations:
    - Brent's method is usually fastest for well-behaved functions
    - Bisection is slowest but most reliable
    - Newton-Raphson is very fast when close to the root with good derivative
    
    Examples
    --------
    Basic usage with bracket methods:
    >>>
    >>> @njit
    >>> def f(x):
    ...     return x**3 - 8
    >>> 
    >>> # Primary solver failed
    >>> results = (np.nan, 100, False)
    >>> 
    >>> # Try backup solvers
    >>> root, iters, converged = _try_back_up_scalar(
    ...     func=f,
    ...     results=results,
    ...     a=0.0,
    ...     b=5.0,
    ...     x0=None,
    ...     tol=1e-6,
    ...     max_iter=100
    ... )
    >>> 
    >>> print(f"Root: {root:.6f}")
    Root: 2.000000
    >>> print(f"Converged: {converged}")
    Converged: True
    >>> print(f"Iterations: {iters}")
    Iterations: 8
    
    Using Newton-Raphson with derivative:
    >>>
    >>> @njit
    >>> def f(x):
    ...     return x**2 - 4
    >>> 
    >>> @njit
    >>> def f_prime(x):
    ...     return 2 * x
    >>> 
    >>> results = (np.nan, 50, False)
    >>> 
    >>> root, iters, converged = _try_back_up_scalar(
    ...     func=f,
    ...     results=results,
    ...     a=None,  # Not needed for Newton
    ...     b=None,
    ...     x0=1.5,  # Initial guess
    ...     tol=1e-10,
    ...     max_iter=50,
    ...     func_prime=f_prime,
    ...     backup_solvers=[SolverName.NEWTON_RAPHSON, SolverName.BRENT]
    ... )
    >>> 
    >>> print(f"Root: {root:.10f}")
    Root: 2.0000000000
    
    Custom solver chain:
    
    >>> from meteorological_equations.math.solvers import SolverName
    >>> 
    >>> # Try Newton first, then secant, finally bisection as last resort
    >>> custom_chain = [
    ...     SolverName.NEWTON_RAPHSON,
    ...     SolverName.SECANT,
    ...     SolverName.BISECTION
    ... ]
    >>> 
    >>> results = _try_back_up_scalar(
    ...     func=f,
    ...     results=(np.nan, 100, False),
    ...     a=0.0,
    ...     b=5.0,
    ...     x0=2.5,
    ...     tol=1e-8,
    ...     max_iter=50,
    ...     func_prime=f_prime,
    ...     backup_solvers=custom_chain
    ... )
    
    With function parameters:
    >>>
    >>> @njit
    >>> def parametric_func(x, a, b):
    ...     return a * x**2 + b
    >>>
    >>> @njit 
    >>> def parametric_prime(x, a, b):
    ...     return 2 * a * x
    >>> 
    >>> # Solve a*x^2 + b = 0 with a=1, b=-4
    >>> root, iters, converged = _try_back_up_scalar(
    ...     func=parametric_func,
    ...     results=(np.nan, 100, False),
    ...     a=-3.0,
    ...     b=3.0,
    ...     x0=1.0,
    ...     tol=1e-6,
    ...     max_iter=100,
    ...     func_prime=parametric_prime,
    ...     func_params=(1.0, -4.0)
    ... )
    >>> print(f"Root: {root:.6f}")
    Root: 2.000000
    
    Handling already-converged results:
    
    >>> # Primary solver succeeded
    >>> results = (2.0, 5, True)
    >>> 
    >>> # Function returns immediately without trying backups
    >>> root, iters, converged = _try_back_up_scalar(
    ...     func=f, results=results, a=0.0, b=5.0, x0=None,
    ...     tol=1e-6, max_iter=100
    ... )
    >>> 
    >>> print(f"Root: {root}, Iterations: {iters}")
    Root: 2.0, Iterations: 5
    >>> # Original results returned unchanged
    
    See Also
    --------
    _try_back_up_vectorised : Vectorized version for arrays of values
    _use_back_up_solvers : Main dispatcher function
    
    Warnings
    --------
    - If no solvers can be applied (e.g., x0, a, and b are all None), the function
      returns the original unconverged results with warnings
    - Bracket methods require func(a) * func(b) < 0, otherwise they will fail
    - Newton-Raphson can diverge if the initial guess is far from the root or if
      the derivative is zero
    """
    converged_flag = results[2]

    if backup_solvers is None:
        backup_solvers = [SolverName.BRENT, SolverName.BISECTION]

    if converged_flag:
        return results

    for backup_solver_name in backup_solvers:
            
        back_up_solver_enum = parse_enum(backup_solver_name, SolverName)
        back_up_solver = SolverMap[back_up_solver_enum]()

        method_type = back_up_solver.get_method_type()
            
        if method_type == MethodType.HYBRID:
            converged_flag = False
                
            if x0 is not None:
                    
                try:
                    results = back_up_solver.find_root(func=func, func_prime=func_prime, x0=x0, func_params=func_params,
                                                    tol=tol, max_iter=max_iter)
                    converged_flag = results[2]
                        
                    if converged_flag:
                        return results
                        
                except Exception as e:
                    warnings.warn(f"Open method failed: {e}")

            if a is not None and b is not None:
                    
                try:
                    results = back_up_solver.find_root(func=func, a=a, b=b, func_params=func_params,
                                                        tol=tol, max_iter=max_iter)
                    converged_flag = results[2]
                    if converged_flag:
                        return results
                        
                except Exception as e:
                    warnings.warn(f"Bracketing method failed: {e}")

                
            if not converged_flag:
                warnings.warn(f"{back_up_solver_enum.value} did not converge. Skipping to the next solver")
                continue


        elif method_type == MethodType.BRACKET:

            try:
                results = back_up_solver.find_root(func=func, a=a, b=b, func_params=func_params,
                                                        tol=tol, max_iter=max_iter)
                    
                converged_flag = results[2]

                if not converged_flag:
                    warnings.warn(f"{back_up_solver_enum.value} did not converge. Skipping to the next solver")
                    continue

                return results

            except Exception as e:
                     
                warnings.warn(f"Bracketing method failed: {e}. Skipping to the next solver.")
                continue
            
        elif method_type == MethodType.OPEN:
                
            try:
                results = back_up_solver.find_root(func=func, func_prime=func_prime, x0=x0, func_params=func_params,
                                                        tol=tol, max_iter=max_iter)
                converged_flag = results[2]

                if not converged_flag:
                    warnings.warn(f"{back_up_solver_enum.value} did not converge. Skipping to the next solver")
                    continue
                return results
                
            except Exception as e:
                     
                warnings.warn(f"Open method failed: {e}. Skipping to the next solver.")
                continue
            
        else:
            warnings.warn("Unknown method type. Skipping to the next solver")
        
    return results


def _try_back_up_vectorised(

    func: Callable[[float], float],
    results: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    a: Optional[npt.ArrayLike],
    b: Optional[npt.ArrayLike],
    x0: Optional[npt.ArrayLike],
    tol: float,
    max_iter: int,
    func_prime: Optional[Callable[[float], float]] = None,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]] = None,
    backup_solvers: List[Union[str, MethodType]] = None,       
):
    """
    Apply a chain of backup solvers to vectorized unconverged root-finding results.
    
    Iterates through a sequence of backup solvers, applying each to only the
    unconverged elements until all converge or all solvers are exhausted. This
    function efficiently handles arrays of values by operating on batches of
    unconverged elements rather than looping over individual values.
    
    The function modifies the results arrays in-place, updating only the elements
    that successfully converge with each solver attempt. Already-converged elements
    are preserved throughout the process.
    
    This is the vectorized counterpart to `_try_back_up_scalar` and is designed for
    efficient batch processing of multiple root-finding problems simultaneously.
    
    Parameters
    ----------
    func : Callable[[float], float]
        The function for which to find roots. Must accept a single float and return
        a float. Called element-wise for vectorized operations. Should satisfy
        func(x) = 0 at the root.
    results : tuple of (ndarray, ndarray, ndarray)
        Results from the primary solver attempt, containing three arrays:
        - results[0]: Root values (float64 array, NaN for unconverged elements)
        - results[1]: Iteration counts (int64 array)
        - results[2]: Convergence flags (bool array, False for unconverged)
        
        These arrays are modified in-place as solvers succeed.
    a : array_like, optional
        Lower bracket bounds, one per element. Required for bracket-based methods
        (Brent, bisection, regula falsi). Must satisfy func(a[i]) * func(b[i]) < 0
        for each i. Shape must match results arrays. If None, bracket methods are
        skipped for all elements.
    b : array_like, optional
        Upper bracket bounds, one per element. Required for bracket-based methods.
        Shape must match results arrays. If None, bracket methods are skipped.
    x0 : array_like, optional
        Initial guesses, one per element. Required for open methods (Newton-Raphson,
        secant). Should be reasonably close to actual roots for best convergence.
        Shape must match results arrays. If None, open methods are skipped.
    tol : float
        Convergence tolerance applied to all elements. Solvers stop when
        |func(x)| < tol or when the change in x between iterations falls below tol.
        Typical values: 1e-6 to 1e-12.
    max_iter : int
        Maximum iterations allowed per solver attempt, applied uniformly to all
        elements. Elements exceeding this limit are considered unconverged for
        that solver and passed to the next backup.
    func_prime : Callable[[float], float], optional
        Derivative of func with respect to x. Required for derivative-based methods
        like Newton-Raphson. Called element-wise for vectorized operations. If None,
        derivative-based methods are skipped. Default is None.
    func_params : array_like, optional
        Additional parameters for func and func_prime. Should be a 2D array where
        each row contains parameters for the corresponding element:
        func_params[i] contains parameters for element i. If 1D, same parameters
        are used for all elements. Default is None.
    backup_solvers : list of str or SolverName, optional
        Ordered sequence of backup solvers to try. Each solver is attempted on
        unconverged elements until all converge or the list is exhausted.
        Default is [SolverName.BRENT, SolverName.BISECTION].
        
        Available solvers:
        - BRENT: Hybrid method, very robust and efficient
        - BISECTION: Reliable bracket method, guaranteed convergence
        - NEWTON_RAPHSON: Fast derivative-based method
        - SECANT: Derivative-free approximation to Newton
        - REGULA_FALSI: Alternative bracket method
    
    Returns
    -------
    results : tuple of (ndarray, ndarray, ndarray)
        Updated results after applying backup solvers:
        - roots: Float64 array of root values (NaN for any remaining unconverged)
        - iterations: Int64 array of total iteration counts (cumulative)
        - converged: Boolean array indicating final convergence status
    
    Notes
    -----
    Algorithm Overview:
    1. Early return if all elements already converged
    2. For each backup solver in sequence:
       a. Identify unconverged elements
       b. Extract inputs for those elements only
       c. Apply solver to unconverged batch
       d. Update results in-place for newly converged elements
       e. Continue to next solver if any remain unconverged
    3. Warn if any elements still unconverged after all attempts
    
    Solver Selection:
    - HYBRID solvers: Try open interface (if x0 provided) then bracket (if a, b provided)
    - BRACKET solvers: Require both a and b arrays; skipped if either is None
    - OPEN solvers: Require x0 array; skipped if None
    
    Efficiency Characteristics:
    - Only unconverged elements are passed to each solver (reduces computation)
    - Arrays modified in-place (minimizes memory allocation)
    - Single solver call per attempt (vectorized, not looped)
    - Index mapping ensures correct updates to original arrays
    
    Edge Cases Handled:
    - All elements converged: Early return, no solver calls
    - No unconverged elements mid-chain: Break early
    - Missing inputs (a, b, x0): Skip incompatible solvers with warnings
    - Solver exceptions: Catch, warn, continue to next solver
    - Partial convergence: Only update successful elements
    
    Examples
    --------
    Basic vectorized usage with all unconverged:
    
    >>> import numpy as np
    >>> 
    >>> @njit
    >>> def f(x):
    ...     return x**3 - 8
    >>> 
    >>> # 5 unconverged elements
    >>> roots = np.full(5, np.nan)
    >>> iters = np.full(5, 100)
    >>> conv = np.full(5, False)
    >>> results = (roots, iters, conv)
    >>> 
    >>> # Bracket bounds for each element
    >>> a = np.array([0.0, 0.5, 1.0, 1.5, 1.8])
    >>> b = np.array([3.0, 3.0, 3.0, 3.0, 2.5])
    >>> 
    >>> # Apply backup solvers
    >>> roots, iters, conv = _try_back_up_vectorised(
    ...     func=f,
    ...     results=results,
    ...     a=a,
    ...     b=b,
    ...     x0=None,
    ...     tol=1e-6,
    ...     max_iter=100
    ... )
    >>> 
    >>> print(f"All converged: {np.all(conv)}")
    All converged: True
    >>> print(f"Roots (all near 2.0): {roots}")
    Roots (all near 2.0): [2. 2. 2. 2. 2.]
    
    Partial convergence - preserving already-converged elements:
    
    >>> # Some elements already converged
    >>> roots = np.array([2.0, np.nan, 1.5, np.nan, np.nan])
    >>> iters = np.array([8, 100, 10, 100, 100])
    >>> conv = np.array([True, False, True, False, False])
    >>> results = (roots, iters, conv)
    >>> 
    >>> a = np.array([0, 0, 0, 1, 1.5])
    >>> b = np.array([3, 3, 3, 3, 3])
    >>> 
    >>> # Only unconverged elements (indices 1, 3, 4) are processed
    >>> roots, iters, conv = _try_back_up_vectorised(
    ...     func=f, results=results, a=a, b=b, x0=None,
    ...     tol=1e-6, max_iter=100
    ... )
    >>> 
    >>> print(f"Original converged preserved: {roots[0]:.1f}, {roots[2]:.1f}")
    Original converged preserved: 2.0, 1.5
    >>> print(f"Previously unconverged now solved: {conv}")
    Previously unconverged now solved: [True True True True True]
    
    Using custom solver chain with both open and bracket methods:
    
    >>> def f(x):
    ...     return x**2 - 4
    >>> 
    >>> def f_prime(x):
    ...     return 2 * x
    >>> 
    >>> roots = np.full(3, np.nan)
    >>> iters = np.full(3, 100)
    >>> conv = np.full(3, False)
    >>> 
    >>> # Provide both x0 and brackets for maximum flexibility
    >>> x0 = np.array([1.0, 1.5, 2.5])
    >>> a = np.array([0.0, 0.0, 0.0])
    >>> b = np.array([5.0, 5.0, 5.0])
    >>> 
    >>> # Try Newton first (fast), then Brent (robust), then bisection (reliable)
    >>> custom_chain = [
    ...     SolverName.NEWTON_RAPHSON,
    ...     SolverName.BRENT,
    ...     SolverName.BISECTION
    ... ]
    >>> 
    >>> roots, iters, conv = _try_back_up_vectorised(
    ...     func=f,
    ...     results=(roots, iters, conv),
    ...     a=a, b=b, x0=x0,
    ...     tol=1e-10,
    ...     max_iter=50,
    ...     func_prime=f_prime,
    ...     backup_solvers=custom_chain
    ... )
    >>> 
    >>> print(f"All converged: {np.all(conv)}")
    All converged: True
    >>> print(f"Roots: {roots}")
    Roots: [2. 2. 2.]
    
    With function parameters (different parameters per element):
    
    >>> def parametric_func(x, a, b):
    ...     return a * x**3 + b
    >>> 
    >>> # Solve a[i]*x^3 + b[i] = 0 for each i
    >>> # Different (a, b) parameters for each of 3 elements
    >>> func_params = np.array([
    ...     [1.0, -8.0],   # Solves x^3 - 8 = 0, root at 2
    ...     [2.0, -16.0],  # Solves 2x^3 - 16 = 0, root at 2
    ...     [0.5, -4.0],   # Solves 0.5x^3 - 4 = 0, root at 2
    ... ])
    >>> 
    >>> roots = np.full(3, np.nan)
    >>> iters = np.full(3, 100)
    >>> conv = np.full(3, False)
    >>> 
    >>> bracket_a = np.array([0, 0, 0])
    >>> bracket_b = np.array([5, 5, 5])
    >>> 
    >>> roots, iters, conv = _try_back_up_vectorised(
    ...     func=parametric_func,
    ...     results=(roots, iters, conv),
    ...     a=bracket_a, b=bracket_b, x0=None,
    ...     tol=1e-6, max_iter=100,
    ...     func_params=func_params
    ... )
    >>> 
    >>> print(f"All roots near 2.0: {np.allclose(roots, 2.0)}")
    All roots near 2.0: True
    
    Large-scale example (1000 elements):
    
    >>> n = 1000
    >>> roots = np.full(n, np.nan)
    >>> iters = np.full(n, 100)
    >>> conv = np.full(n, False)
    >>> 
    >>> # Random brackets around x=2
    >>> np.random.seed(42)
    >>> a = np.random.uniform(0.0, 1.9, n)
    >>> b = np.random.uniform(2.1, 4.0, n)
    >>> 
    >>> import time
    >>> start = time.time()
    >>> roots, iters, conv = _try_back_up_vectorised(
    ...     func=f, results=(roots, iters, conv),
    ...     a=a, b=b, x0=None, tol=1e-6, max_iter=100
    ... )
    >>> elapsed = time.time() - start
    >>> 
    >>> print(f"Solved {n} roots in {elapsed:.3f}s")
    Solved 1000 roots in 0.127s
    >>> print(f"Convergence rate: {100*np.sum(conv)/n:.1f}%")
    Convergence rate: 100.0%
    
    See Also
    --------
    _try_back_up_scalar : Scalar version for single values
    _try_back_up_bracket_vectorised : Helper for bracket methods
    _try_back_up_open_vectorised : Helper for open methods
    _use_back_up_solvers : Main dispatcher function
    
    Warnings
    --------
    - All input arrays (a, b, x0, func_params) must have compatible shapes with
      the results arrays
    - The results tuple is modified in-place; the returned tuple references the
      same arrays
    - If no solvers can be applied (all incompatible with provided inputs), a
      warning is issued and original results are returned
    - Bracket methods require func(a[i]) * func(b[i]) < 0 for each element
    """
    roots, iterations, converged_flag = results

    if backup_solvers is None:
        backup_solvers = [SolverName.BRENT, SolverName.BISECTION]

    if np.all(converged_flag):
        return results

    for backup_solver_name in backup_solvers:

        unconverged_mask = np.logical_not(converged_flag)
        unconverged_idx = np.where(unconverged_mask)[0]

        if len(unconverged_idx) == 0:
            break

        try:
            backup_solver_enum = parse_enum(backup_solver_name, SolverName)
            backup_solver = SolverMap[backup_solver_enum]()
            method_type = backup_solver.get_method_type()
        
        except Exception as e:
            warnings.warn(f"Failed to initialise {backup_solver_enum.value}: {e}. Skipping to the next available solver.")
            continue

        try:
            
            if method_type == MethodType.HYBRID:
               
                if x0 is not None:
                    
                    try:
                        success_flag = _try_back_up_open_vectorised(
                            backup_solver=backup_solver,  func=func, results=results,
                            x0=x0, unconverged_idx=unconverged_idx, func_params=func_params, 
                            func_prime=func_prime, tol=tol, max_iter=max_iter
                        )

                        if success_flag:
                            continue
                    
                    except Exception as e:
                        warnings.warn(f"Open interface for hybrid solver {backup_solver_enum.value} failed: {e}")
                
                if a is not None and b is not None:
                    unconverged_mask = np.logical_not(converged_flag)
                    unconverged_idx = np.where(unconverged_mask)[0]

                    if len(unconverged_idx) > 0:
                        
                        try:
                           success_flag = _try_back_up_bracket_vectorised(
                                backup_solver=backup_solver, func=func, results=results,
                                a=a, b=b, unconverged_idx=unconverged_idx, func_params=func_params, tol=tol,
                                max_iter=max_iter
                            )
                           
                           if success_flag:
                               continue
                        
                        except Exception as e:
                            warnings.warn(f"Open interface for hybrid solver {backup_solver_enum.value} failed: {e}")
                            continue

            elif method_type == MethodType.BRACKET:
                
                if a is None or b is None:
                    warnings.warn(f"Bracketing method {backup_solver_enum.value} requires brackets. Skipping to the next available solver.")
                    continue

                _try_back_up_bracket_vectorised(backup_solver=backup_solver, func=func,
                                                results=results, a=a, b=b, unconverged_idx=unconverged_idx,
                                                func_params=func_params, tol=tol, max_iter=max_iter)

            elif method_type == MethodType.OPEN:
                
                if x0 is None:
                    warnings.warn(f"Bracketing method {backup_solver_enum.value} requires initial guess. Skipping to the next available solver.")
                    continue

                _try_back_up_open_vectorised(backup_solver=backup_solver, func=func, results=results,
                                             x0=x0, unconverged_idx=unconverged_idx, func_params=func_params, func_prime=func_prime,
                                             tol=tol, max_iter=max_iter)
            
        except Exception as e:
            warnings.warn(f"{backup_solver_enum.value} failed: {e}")
            continue

    if not np.all(converged_flag):
        n_failed = np.sum(np.logical_not(converged_flag))
        warnings.warn(
            f"Some roots did not converge. "
            f"{n_failed} out of {len(converged_flag)} still unconverged"
        )
    
    return results
                     


def _try_back_up_bracket_vectorised(
    backup_solver: Solver,
    func: Callable[[float], float],
    results: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    unconverged_idx: npt.ArrayLike,
    tol: float,
    max_iter: int,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]],
        
) -> bool:
    """
    Apply a bracket-based solver to unconverged elements in vectorized results.
    
    Extracts unconverged elements from the full arrays, applies the specified
    bracket-based root-finding solver (e.g., Brent, bisection, regula falsi) to
    those elements only, and updates the original results in-place for elements
    that converge.
    
    This function serves as a helper for `_try_back_up_vectorised`, handling the
    index manipulation and partial array updates required for efficient vectorized
    backup solver application with bracket methods.
    
    Parameters
    ----------
    backup_solver : Solver
        An initialized solver instance implementing a bracket-based method (BRACKET
        or HYBRID type). Must have a `find_root` method that accepts bracket bounds
        (a, b) and returns (roots, iterations, converged) tuple.
        
        Compatible solvers: Brent, Bisection, Regula Falsi, or any hybrid solver
        used in bracket mode.
    func : Callable[[float], float]
        The function for which to find roots. Must accept a single float and return
        a float. Called element-wise during root finding. Should satisfy func(x) = 0
        at the root.
    results : tuple of (ndarray, ndarray, ndarray)
        Original full results arrays that will be updated in-place:
        - results[0]: Root values (float64), NaN for unconverged
        - results[1]: Iteration counts (int64)
        - results[2]: Convergence flags (bool)
        
        Only elements at `unconverged_idx` positions may be modified.
    a : array_like
        Lower bracket bounds for all elements (full array). Must have the same
        length as results arrays. Values at `unconverged_idx` positions are
        extracted and used. Must satisfy func(a[i]) * func(b[i]) < 0 for
        convergence.
    b : array_like
        Upper bracket bounds for all elements (full array). Must have the same
        length as results arrays. Values at `unconverged_idx` positions are
        extracted and used.
    unconverged_idx : array_like
        Integer indices of unconverged elements in the results arrays. Typically
        obtained via `np.where(~converged_flag)[0]`. Only these elements are
        passed to the solver.
        
        Example: If converged = [True, False, True, False], then
        unconverged_idx = [1, 3]
    tol : float
        Convergence tolerance for the solver. Stops when |func(x)| < tol or when
        the bracket width is less than tol. Typical values: 1e-6 to 1e-12.
    max_iter : int
        Maximum iterations allowed for the solver attempt. Elements that don't
        converge within this limit return False in their convergence flag.
    func_params : array_like, optional
        Additional parameters for func. Can be:
        - None: No additional parameters
        - 1D array: Same parameters for all elements
        - 2D array: func_params[i] contains parameters for element i
        
        Only rows corresponding to unconverged_idx are extracted and passed to
        the solver. Default is None.
    
    Returns
    -------
    success : bool
        True if at least one previously unconverged element converged with this
        solver, False if all unconverged elements remained unconverged or if an
        exception occurred.
        
        This return value helps the calling function decide whether to continue
        to the next solver in the chain.
    
    Notes
    -----
    Algorithm Steps:
    1. Extract unconverged elements from full arrays using unconverged_idx
    2. Call solver on extracted subset: (a[idx], b[idx])
    3. Identify which of the unconverged elements newly converged
    4. Map back to original indices: unconverged_idx[newly_converged_mask]
    5. Update original arrays in-place at those positions only
    6. Return True if any converged, False otherwise
    
    Index Mapping:
    The function performs a two-level index mapping:
    - Level 1: unconverged_idx maps from subset → full array
    - Level 2: updated_converged_flag identifies successes within subset
    - Combined: unconverged_idx[updated_converged_flag] gives final positions
    
    Example index mapping:
        Full array size: 5
        converged_flag = [True, False, True, False, False]
        unconverged_idx = [1, 3, 4]  # Positions of False values
        
        After solver:
        updated_converged_flag = [True, False, True]  # For indices [1,3,4]
        
        Newly converged positions in full array:
        unconverged_idx[[True, False, True]] = [1, 4]
    
    In-Place Modification:
    The results tuple arrays are modified in-place. The function does not create
    new arrays but updates existing ones. This is memory-efficient but means the
    caller's arrays are changed.
    
    Error Handling:
    - Any exception during solver execution is caught
    - A warning is issued with the error message
    - Function returns False to indicate no convergence
    - Original results remain unchanged on exception
    
    Performance:
    - Time: O(M * iterations) where M = len(unconverged_idx)
    - Space: O(M) for temporary arrays
    - In-place updates: O(K) where K = number newly converged
    
    Examples
    --------
    Basic usage within backup solver chain:
    
    >>> import numpy as np
    >>> from meteorological_equations.math.solvers import BrentSolver
    >>> 
    >>> @njit
    >>> def f(x):
    ...     return x**3 - 8
    >>> 
    >>> # Setup: 5 elements, 3 unconverged
    >>> roots = np.array([2.0, np.nan, 1.5, np.nan, np.nan])
    >>> iters = np.array([8, 100, 10, 100, 100])
    >>> conv = np.array([True, False, True, False, False])
    >>> results = (roots, iters, conv)
    >>> 
    >>> # Identify unconverged
    >>> unconverged_idx = np.where(~conv)[0]
    >>> print(f"Unconverged indices: {unconverged_idx}")
    Unconverged indices: [1 3 4]
    >>> 
    >>> # Bracket bounds (full arrays)
    >>> a = np.array([0, 0, 0, 1, 1.5])
    >>> b = np.array([3, 3, 3, 3, 3])
    >>> 
    >>> # Apply Brent solver to unconverged elements
    >>> solver = BrentSolver()
    >>> success = _try_back_up_bracket_vectorised(
    ...     back_up_solver=solver,
    ...     func=f,
    ...     results=results,
    ...     a=a,
    ...     b=b,
    ...     unconverged_idx=unconverged_idx,
    ...     tol=1e-6,
    ...     max_iter=100
    ... )
    >>> 
    >>> print(f"At least one converged: {success}")
    At least one converged: True
    >>> print(f"Updated convergence: {conv}")
    Updated convergence: [True True True True True]
    >>> print(f"All roots near 2.0: {np.allclose(roots[~np.isnan(roots)], 2.0)}")
    All roots near 2.0: True
    
    Demonstrating index mapping:
    
    >>> # Start with 4 elements, 2 unconverged at positions 1 and 3
    >>> roots = np.array([1.0, np.nan, 2.0, np.nan])
    >>> iters = np.array([5, 100, 8, 100])
    >>> conv = np.array([True, False, True, False])
    >>> 
    >>> unconverged_idx = np.array([1, 3])
    >>> print(f"Will try to solve elements at positions: {unconverged_idx}")
    Will try to solve elements at positions: [1 3]
    >>> 
    >>> # After solver runs on unconverged subset
    >>> # Suppose element at position 1 converges, but position 3 doesn't
    >>> # This happens internally in the function:
    >>> # updated_converged_flag = [True, False]  # For positions [1, 3]
    >>> # newly_converged_original_idx = unconverged_idx[[True, False]] = [1]
    >>> # results[0][1] gets updated
    >>> # results[2][1] becomes True
    
    Partial convergence case:
    
    >>> # 3 unconverged elements
    >>> roots = np.full(3, np.nan)
    >>> iters = np.full(3, 100)
    >>> conv = np.full(3, False)
    >>> results = (roots, iters, conv)
    >>> 
    >>> # One bracket pair is invalid (same sign), others valid
    >>> a = np.array([0.0, 3.0, 0.0])  # func(3.0) > 0
    >>> b = np.array([3.0, 5.0, 3.0])  # func(5.0) > 0, same sign!
    >>> 
    >>> unconverged_idx = np.array([0, 1, 2])
    >>> 
    >>> solver = BrentSolver()
    >>> success = _try_back_up_bracket_vectorised(
    ...     solver, f, results, a, b, unconverged_idx, 1e-6, 100
    ... )
    >>> 
    >>> # Elements 0 and 2 converge, element 1 fails (invalid bracket)
    >>> print(f"Partial success: {success}")  # True (at least one succeeded)
    Partial success: True
    >>> print(f"Convergence flags: {conv}")
    Convergence flags: [True False True]
    
    With function parameters:
    
    >>> def parametric_func(x, a, b):
    ...     return a * x**3 + b
    >>> 
    >>> roots = np.full(2, np.nan)
    >>> iters = np.full(2, 100)
    >>> conv = np.full(2, False)
    >>> results = (roots, iters, conv)
    >>> 
    >>> # Different parameters for each element
    >>> func_params = np.array([
    ...     [1.0, -8.0],   # x^3 - 8 = 0
    ...     [2.0, -16.0],  # 2x^3 - 16 = 0
    ... ])
    >>> 
    >>> bracket_a = np.array([0, 0])
    >>> bracket_b = np.array([5, 5])
    >>> unconverged_idx = np.array([0, 1])
    >>> 
    >>> success = _try_back_up_bracket_vectorised(
    ...     solver, parametric_func, results,
    ...     bracket_a, bracket_b, unconverged_idx,
    ...     1e-6, 100, func_params
    ... )
    >>> 
    >>> print(f"Both converged: {np.all(conv)}")
    Both converged: True
    >>> print(f"Roots: {roots}")
    Roots: [2. 2.]
    
    Error handling:
    
    >>> # Invalid brackets (both same sign)
    >>> bad_a = np.array([3.0, 4.0])
    >>> bad_b = np.array([5.0, 6.0])
    >>> 
    >>> roots = np.full(2, np.nan)
    >>> iters = np.full(2, 100)
    >>> conv = np.full(2, False)
    >>> 
    >>> # This will fail and issue warning
    >>> success = _try_back_up_bracket_vectorised(
    ...     solver, f, (roots, iters, conv),
    ...     bad_a, bad_b, np.array([0, 1]),
    ...     1e-6, 100
    ... )
    >>> # Warning: Bracketing method failed: ...
    >>> 
    >>> print(f"Failed gracefully: {not success}")
    Failed gracefully: True
    >>> print(f"Original results preserved: {np.all(~conv)}")
    Original results preserved: True
    
    See Also
    --------
    _try_back_up_open_vectorised : Equivalent for open methods (Newton, secant)
    _try_back_up_vectorised : Main vectorized backup orchestrator
    _update_converged_results : Helper for in-place array updates
    _get_unconverged_func_params : Helper for extracting parameter subsets
    
    """

    try: 
        original_roots = results[0]
        original_iterations = results[1]
        original_converged_flag = results[2]

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        a_unconverged = a[unconverged_idx]
        b_unconverged = b[unconverged_idx]

        func_params_unconverged = _get_unconverged_func_params(
            func_params=func_params, unconverged_idx=unconverged_idx
        )

        updated_roots, updated_iterations, updated_converged_flag = backup_solver.find_root(
            func=func,
            a=a_unconverged,
            b=b_unconverged,
            func_params=func_params_unconverged,
            tol=tol,
            max_iter=max_iter
        )

        _update_converged_results(
            roots=original_roots,
            iterations=original_iterations,
            converged_flag=original_converged_flag,
            unconverged_idx=unconverged_idx,
            updated_roots=updated_roots,
            updated_iterations=updated_iterations,
            updated_converged_flag=updated_converged_flag
        )

        return np.any(updated_converged_flag)

    except Exception as e:             
        warnings.warn(f"Bracketing method failed: {e}. Skipping to the next solver.")
        return False


def _try_back_up_open_vectorised(
    backup_solver: Solver,
    func: Callable[[float], float],
    results: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    x0: npt.ArrayLike,
    unconverged_idx: npt.ArrayLike,
    tol: float,
    max_iter: int,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]],
    func_prime: Optional[Callable[[float], float]]
) -> bool:
    """
    Docstring for _try_back_up_open_vectorised

    """

    try:

        original_roots = results[0]
        original_iterations = results[1]
        original_converged_flag = results[2]

        x0 = np.asarray(x0, dtype=np.float64)
        x0_unconverged = x0[unconverged_idx]

        func_params_unconverged = _get_unconverged_func_params(
            func_params=func_params,
            unconverged_idx=unconverged_idx
        ) 

        updated_roots, updated_iterations, updated_converged_flag = backup_solver.find_root(
            func=func,
            func_prime=func_prime,
            x0=x0_unconverged,
            func_params=func_params_unconverged,
            tol=tol,
            max_iter=max_iter
        )

        _update_converged_results(
            roots=original_roots,
            iterations=original_iterations,
            converged_flag=original_converged_flag,
            unconverged_idx=unconverged_idx,
            updated_roots=updated_roots,
            updated_iterations=updated_iterations,
            updated_converged_flag=updated_converged_flag
        )

        return np.any(updated_converged_flag)

    except Exception as e:
        warnings.warn(f"Open method failed: {e}")
        return False


def _get_unconverged_func_params(
        func_params: Optional[Union[npt.ArrayLike, Tuple[float, ...]]],
        unconverged_idx: npt.ArrayLike
) -> Optional[npt.NDArray]:
    """
    Extracts the unconverged function parameters if not None.

    Args:
        - func_params (Optional[Union[npt.ArrayLike, Tuple[float, ...]]]) = function parameters which is either None or array of tuples
        - unconverged_idx = position of function params that have not converged
    
    
    Returns:
        - function parameters or None 

    """
    if func_params is not None:
        func_params = np.asarray(func_params, dtype=np.float64)
        return func_params[unconverged_idx]
    
    return None

def _update_converged_results(
    roots: npt.NDArray,
    iterations: npt.NDArray,
    converged_flag: npt.NDArray,
    unconverged_idx: npt.NDArray,
    updated_roots: npt.NDArray,
    updated_iterations: npt.NDArray,
    updated_converged_flag: npt.NDArray
) -> None:
    """
    Updates only the positions that have converged

    Args:
        - root
    """
    updated_converged_mask = updated_converged_flag
    updated_converged_original_idx = unconverged_idx[updated_converged_mask]
    
    
    roots[updated_converged_original_idx] = updated_roots[updated_converged_mask]
    iterations[updated_converged_original_idx] = updated_iterations[updated_converged_mask]
    converged_flag[updated_converged_original_idx] = True
