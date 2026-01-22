"""
Logic for using backup solvers.

Author: Cian Quezon
"""

import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from meteorological_equations.math.solvers._enums import MethodType, SolverName
from meteorological_equations.math.solvers._solvers import Solver
from meteorological_equations.meteorological_equations.math.solvers._types import SolverMap
from meteorological_equations.shared._enum_tools import parse_enum


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
    backup_solvers: List[Union[str, MethodType]] = None,
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
            backup_solvers=backup_solvers,
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
            backup_solvers=backup_solvers,
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
                    results = back_up_solver.find_root(
                        func=func,
                        func_prime=func_prime,
                        x0=x0,
                        func_params=func_params,
                        tol=tol,
                        max_iter=max_iter,
                    )
                    converged_flag = results[2]

                    if converged_flag:
                        return results

                except Exception as e:
                    warnings.warn(f"Open method failed: {e}", stacklevel=2)

            if a is not None and b is not None:
                try:
                    results = back_up_solver.find_root(
                        func=func, a=a, b=b, func_params=func_params, tol=tol, max_iter=max_iter
                    )
                    converged_flag = results[2]
                    if converged_flag:
                        return results

                except Exception as e:
                    warnings.warn(f"Bracketing method failed: {e}", stacklevel=2)

            if not converged_flag:
                warnings.warn(
                    f"{back_up_solver_enum.value} did not converge. Skipping to the next solver",
                    stacklevel=2,
                )
                continue

        elif method_type == MethodType.BRACKET:
            try:
                results = back_up_solver.find_root(
                    func=func, a=a, b=b, func_params=func_params, tol=tol, max_iter=max_iter
                )

                converged_flag = results[2]

                if not converged_flag:
                    warnings.warn(
                        f"{back_up_solver_enum.value} did not converge. Skipping to the next solver",
                        stacklevel=2,
                    )
                    continue

                return results

            except Exception as e:
                warnings.warn(
                    f"Bracketing method failed: {e}. Skipping to the next solver.", stacklevel=2
                )
                continue

        elif method_type == MethodType.OPEN:
            try:
                results = back_up_solver.find_root(
                    func=func,
                    func_prime=func_prime,
                    x0=x0,
                    func_params=func_params,
                    tol=tol,
                    max_iter=max_iter,
                )
                converged_flag = results[2]

                if not converged_flag:
                    warnings.warn(
                        f"{back_up_solver_enum.value} did not converge. Skipping to the next solver",
                        stacklevel=2,
                    )
                    continue
                return results

            except Exception as e:
                warnings.warn(
                    f"Open method failed: {e}. Skipping to the next solver.", stacklevel=2
                )
                continue

        else:
            warnings.warn("Unknown method type. Skipping to the next solver", stacklevel=2)

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
            warnings.warn(
                f"Failed to initialise {backup_solver_enum.value}: {e}. Skipping to the next available solver.",
                stacklevel=2,
            )
            continue

        try:
            if method_type == MethodType.HYBRID:
                if x0 is not None:
                    try:
                        success_flag = _try_back_up_open_vectorised(
                            backup_solver=backup_solver,
                            func=func,
                            results=results,
                            x0=x0,
                            unconverged_idx=unconverged_idx,
                            func_params=func_params,
                            func_prime=func_prime,
                            tol=tol,
                            max_iter=max_iter,
                        )

                        if success_flag:
                            continue

                    except Exception as e:
                        warnings.warn(
                            f"Open interface for hybrid solver {backup_solver_enum.value} failed: {e}",
                            stacklevel=2,
                        )

                if a is not None and b is not None:
                    unconverged_mask = np.logical_not(converged_flag)
                    unconverged_idx = np.where(unconverged_mask)[0]

                    if len(unconverged_idx) > 0:
                        try:
                            success_flag = _try_back_up_bracket_vectorised(
                                backup_solver=backup_solver,
                                func=func,
                                results=results,
                                a=a,
                                b=b,
                                unconverged_idx=unconverged_idx,
                                func_params=func_params,
                                tol=tol,
                                max_iter=max_iter,
                            )

                            if success_flag:
                                continue

                        except Exception as e:
                            warnings.warn(
                                f"Open interface for hybrid solver {backup_solver_enum.value} failed: {e}",
                                stacklevel=2,
                            )
                            continue

            elif method_type == MethodType.BRACKET:
                if a is None or b is None:
                    warnings.warn(
                        f"Bracketing method {backup_solver_enum.value} requires brackets. Skipping to the next available solver.",
                        stacklevel=2,
                    )
                    continue

                _try_back_up_bracket_vectorised(
                    backup_solver=backup_solver,
                    func=func,
                    results=results,
                    a=a,
                    b=b,
                    unconverged_idx=unconverged_idx,
                    func_params=func_params,
                    tol=tol,
                    max_iter=max_iter,
                )

            elif method_type == MethodType.OPEN:
                if x0 is None:
                    warnings.warn(
                        f"Bracketing method {backup_solver_enum.value} requires initial guess. Skipping to the next available solver.",
                        stacklevel=2,
                    )
                    continue

                _try_back_up_open_vectorised(
                    backup_solver=backup_solver,
                    func=func,
                    results=results,
                    x0=x0,
                    unconverged_idx=unconverged_idx,
                    func_params=func_params,
                    func_prime=func_prime,
                    tol=tol,
                    max_iter=max_iter,
                )

        except Exception as e:
            warnings.warn(f"{backup_solver_enum.value} failed: {e}", stacklevel=2)
            continue

    if not np.all(converged_flag):
        n_failed = np.sum(np.logical_not(converged_flag))
        warnings.warn(
            f"Some roots did not converge. "
            f"{n_failed} out of {len(converged_flag)} still unconverged",
            stacklevel=2,
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
            max_iter=max_iter,
        )

        _update_converged_results(
            roots=original_roots,
            iterations=original_iterations,
            converged_flag=original_converged_flag,
            unconverged_idx=unconverged_idx,
            updated_roots=updated_roots,
            updated_iterations=updated_iterations,
            updated_converged_flag=updated_converged_flag,
        )

        return np.any(updated_converged_flag)

    except Exception as e:
        warnings.warn(f"Bracketing method failed: {e}. Skipping to the next solver.", stacklevel=2)
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
    func_prime: Optional[Callable[[float], float]],
) -> bool:
    """
    Apply an open method solver to unconverged elements in vectorized results.

    Extracts unconverged elements from the full arrays, applies the specified
    open method root-finding solver (e.g., Newton-Raphson, secant) to those
    elements only, and updates the original results in-place for elements that
    converge.

    Open methods use iteration from an initial guess without requiring brackets,
    making them faster than bracket methods when the initial guess is good but
    less robust when far from the root.

    This function serves as a helper for `_try_back_up_vectorised`, handling the
    index manipulation and partial array updates required for efficient vectorized
    backup solver application with open methods.

    Parameters
    ----------
    back_up_solver : Solver
        An initialized solver instance implementing an open method (OPEN or HYBRID
        type). Must have a `find_root` method that accepts an initial guess (x0)
        and optionally a derivative (func_prime), returning (roots, iterations,
        converged) tuple.

        Compatible solvers: Newton-Raphson, Secant, or any hybrid solver used in
        open mode.
    func : Callable[[float], float]
        The function for which to find roots. Must accept a single float and return
        a float. Should be decorated with `@njit` (Numba JIT compilation) for
        compatibility with the solver infrastructure. Called element-wise during
        root finding. Should satisfy func(x) = 0 at the root.
    results : tuple of (ndarray, ndarray, ndarray)
        Original full results arrays that will be updated in-place:
        - results[0]: Root values (float64), NaN for unconverged
        - results[1]: Iteration counts (int64)
        - results[2]: Convergence flags (bool)

        Only elements at `unconverged_idx` positions may be modified.
    x0 : array_like
        Initial guesses for all elements (full array). Must have the same length
        as results arrays. Values at `unconverged_idx` positions are extracted
        and used as starting points for the iteration. Good initial guesses are
        critical for open methods - poor guesses may cause divergence.
    unconverged_idx : array_like
        Integer indices of unconverged elements in the results arrays. Typically
        obtained via `np.where(~converged_flag)[0]`. Only these elements are
        passed to the solver.

        Example: If converged = [True, False, True, False], then
        unconverged_idx = [1, 3]
    tol : float
        Convergence tolerance for the solver. Iteration stops when |func(x)| < tol
        or when |x_new - x_old| < tol. Typical values: 1e-6 to 1e-12.
    max_iter : int
        Maximum iterations allowed for the solver attempt. Elements that don't
        converge within this limit return False in their convergence flag. Open
        methods typically need fewer iterations than bracket methods when
        converging, but may diverge if the initial guess is poor.
    func_params : array_like, optional
        Additional parameters for func and func_prime. Can be:
        - None: No additional parameters
        - 1D array: Same parameters for all elements
        - 2D array: func_params[i] contains parameters for element i

        Only rows corresponding to unconverged_idx are extracted and passed to
        the solver. Default is None.
    func_prime : Callable[[float], float], optional
        Derivative of func with respect to x (df/dx). Required for derivative-based
        methods like Newton-Raphson. Should be decorated with `@njit` for
        compatibility. Called element-wise during iteration. If None, derivative-free
        methods (like secant) must be used. Default is None.

    Returns
    -------
    success : bool
        True if at least one previously unconverged element converged with this
        solver, False if all unconverged elements remained unconverged or if an
        exception occurred.

        This return value helps the calling function decide whether to continue
        to the next solver in the chain.

    See Also
    --------
    _try_back_up_bracket_vectorised : Equivalent for bracket methods
    _try_back_up_vectorised : Main vectorized backup orchestrator
    _update_converged_results : Helper for in-place array updates
    _get_unconverged_func_params : Helper for extracting parameter subsets


    Warnings
    --------
    - Functions (func, func_prime) MUST be decorated with @njit for compatibility
    - Initial guesses should be reasonably close to roots for best convergence
    - Newton-Raphson fails when derivative is zero or near-zero
    - Open methods can diverge - always have bracket methods as fallback
    - All input arrays (x0, func_params) must have compatible shapes with results
    - Results tuple is modified in-place; returned tuple references same arrays
    """

    try:
        original_roots = results[0]
        original_iterations = results[1]
        original_converged_flag = results[2]

        x0 = np.asarray(x0, dtype=np.float64)
        x0_unconverged = x0[unconverged_idx]

        func_params_unconverged = _get_unconverged_func_params(
            func_params=func_params, unconverged_idx=unconverged_idx
        )

        updated_roots, updated_iterations, updated_converged_flag = backup_solver.find_root(
            func=func,
            func_prime=func_prime,
            x0=x0_unconverged,
            func_params=func_params_unconverged,
            tol=tol,
            max_iter=max_iter,
        )

        _update_converged_results(
            roots=original_roots,
            iterations=original_iterations,
            converged_flag=original_converged_flag,
            unconverged_idx=unconverged_idx,
            updated_roots=updated_roots,
            updated_iterations=updated_iterations,
            updated_converged_flag=updated_converged_flag,
        )

        return np.any(updated_converged_flag)

    except Exception as e:
        warnings.warn(f"Open method failed: {e}", stacklevel=2)
        return False


def _get_unconverged_func_params(
    func_params: Optional[Union[npt.ArrayLike, Tuple[float, ...]]], unconverged_idx: npt.ArrayLike
) -> Optional[npt.NDArray]:
    """
    Extract function parameters for unconverged elements only.

    This utility function handles the index-based extraction of function parameters
    corresponding to unconverged elements. It's used by vectorized backup solvers to
    ensure that only the necessary parameter subset is passed to the solver, reducing
    memory usage and computation.

    The function handles the common case where func_params is None (no additional
    parameters needed) and the array case where different parameters apply to
    different elements.

    Parameters
    ----------
    func_params : array_like or tuple of floats, optional
        Function parameters for all elements. Can be:

        - None: No additional parameters (function only depends on x)
        - 1D array/tuple: Same parameters used for all elements, e.g., (a, b)
          Returns the same parameters regardless of unconverged_idx
        - 2D array: Different parameters per element, shape (n_elements, n_params)
          func_params[i, :] contains parameters for element i
          Only rows at unconverged_idx positions are extracted

        These parameters are passed to func as additional arguments:
        func(x, *params) where params comes from this array.
    unconverged_idx : array_like of int
        Integer indices specifying which elements are unconverged. Typically
        obtained from `np.where(~converged_flag)[0]`. These indices are used to
        extract the corresponding rows from func_params.

        Example: If unconverged_idx = [1, 3, 4], only func_params[1],
        func_params[3], and func_params[4] are returned.

    Returns
    -------
    params_subset : ndarray or None
        - None: If func_params was None (no parameters needed)
        - ndarray: Subset of parameters for unconverged elements only
          + 1D case: Returns original parameters unchanged (all elements share them)
          + 2D case: Returns func_params[unconverged_idx], shape (len(unconverged_idx), n_params)

    Examples
    --------
    Case 1: No parameters (None):

    >>> unconverged_idx = np.array([1, 3, 4])
    >>> result = _get_unconverged_func_params(None, unconverged_idx)
    >>> print(result)
    None

    Case 2: Shared parameters (1D) - same for all elements:

    >>> # All elements use (a=2.0, b=-8.0)
    >>> func_params = np.array([2.0, -8.0])
    >>> unconverged_idx = np.array([1, 3, 4])
    >>>
    >>> result = _get_unconverged_func_params(func_params, unconverged_idx)
    >>> print(result)
    [2. -8.]
    >>> print(result.shape)
    (2,)
    >>> # Same parameters returned for all unconverged elements

    Case 3: Per-element parameters (2D) - different for each element:

    >>> # 5 elements, each with (a, b) parameters
    >>> func_params = np.array([
    ...     [1.0, -4.0],   # Element 0: a=1, b=-4
    ...     [2.0, -8.0],   # Element 1: a=2, b=-8
    ...     [3.0, -12.0],  # Element 2: a=3, b=-12
    ...     [1.5, -6.0],   # Element 3: a=1.5, b=-6
    ...     [0.5, -2.0],   # Element 4: a=0.5, b=-2
    ... ])
    >>>
    >>> # Only elements 1, 3, 4 are unconverged
    >>> unconverged_idx = np.array([1, 3, 4])
    >>>
    >>> result = _get_unconverged_func_params(func_params, unconverged_idx)
    >>> print(result)
    [[ 2.  -8. ]
     [ 1.5 -6. ]
     [ 0.5 -2. ]]
    >>> print(result.shape)
    (3, 2)
    >>> # Returns only rows [1, 3, 4] - parameters for unconverged elements

    Realistic usage in backup solver context:

    >>> from numba import njit
    >>>
    >>> @njit
    ... def parametric_func(x, a, b):
    ...     '''Solve a*x^2 + b = 0 for various (a, b) values.'''
    ...     return a * x**2 + b
    >>>
    >>> # 1000 elements with different parameters
    >>> n = 1000
    >>> func_params = np.random.randn(n, 2)
    >>>
    >>> # Only 50 elements unconverged
    >>> converged = np.random.rand(n) > 0.05
    >>> unconverged_idx = np.where(~converged)[0]
    >>> print(f"Unconverged: {len(unconverged_idx)} out of {n}")
    Unconverged: 50 out of 1000
    >>>
    >>> # Extract only what's needed
    >>> params_subset = _get_unconverged_func_params(func_params, unconverged_idx)
    >>> print(f"Original shape: {func_params.shape}")
    Original shape: (1000, 2)
    >>> print(f"Subset shape: {params_subset.shape}")
    Subset shape: (50, 2)
    >>> print(f"Memory saved: {100*(1 - 50/1000):.1f}%")
    Memory saved: 95.0%
    >>>
    >>> # Now only pass 50 parameter sets to solver instead of 1000

    Edge case - empty unconverged_idx:

    >>> func_params = np.array([[1, 2], [3, 4], [5, 6]])
    >>> unconverged_idx = np.array([])  # Empty - all converged
    >>>
    >>> result = _get_unconverged_func_params(func_params, unconverged_idx)
    >>> print(result)
    []
    >>> print(result.shape)
    (0, 2)
    >>> # Returns empty array with correct parameter dimension

    Type conversion from tuple:

    >>> # Input as tuple instead of array
    >>> func_params = (2.0, -8.0)
    >>> unconverged_idx = np.array([0, 1, 2])
    >>>
    >>> result = _get_unconverged_func_params(func_params, unconverged_idx)
    >>> print(type(result))
    <class 'numpy.ndarray'>
    >>> print(result.dtype)
    float64
    >>> # Automatically converted to float64 ndarray

    Integration with solver workflow:

    >>> # Typical usage in _try_back_up_open_vectorised
    >>> def solver_workflow_example():
    ...     # Full arrays
    ...     roots = np.array([2.0, np.nan, 1.5, np.nan, np.nan])
    ...     func_params_full = np.array([
    ...         [1, -4], [2, -8], [3, -12], [1.5, -6], [0.5, -2]
    ...     ])
    ...
    ...     # Find unconverged
    ...     converged = ~np.isnan(roots)
    ...     unconverged_idx = np.where(~converged)[0]
    ...     print(f"Unconverged indices: {unconverged_idx}")
    ...
    ...     # Extract only unconverged parameters
    ...     params_subset = _get_unconverged_func_params(
    ...         func_params_full, unconverged_idx
    ...     )
    ...     print(f"Parameters for unconverged: {params_subset}")
    ...
    ...     # Pass to solver - only 3 parameter sets instead of 5
    ...     # result = solver.find_root(..., func_params=params_subset)
    >>>
    >>> solver_workflow_example()
    Unconverged indices: [1 3 4]
    Parameters for unconverged: [[ 2.  -8. ]
     [ 1.5 -6. ]
     [ 0.5 -2. ]]

    See Also
    --------
    _update_converged_results : Counterpart function for updating results
    _try_back_up_open_vectorised : Uses this for open method parameters
    _try_back_up_bracket_vectorised : Uses this for bracket method parameters

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
    updated_converged_flag: npt.NDArray,
) -> None:
    """
    Update original result arrays in-place with newly converged elements.

    Performs selective in-place updates of the original full-size result arrays,
    replacing only the elements that newly converged during a backup solver attempt.
    This function handles the critical index mapping from the unconverged subset
    back to the original array positions.

    The function modifies the input arrays directly (in-place) and returns None,
    following NumPy conventions for in-place operations. This approach is memory-
    efficient and allows the calling function to see updates immediately.

    Parameters
    ----------
    roots : ndarray
        Original full-size array of root values (float64). Contains roots from
        previous solver attempts, with NaN for unconverged elements. Modified
        in-place to include newly converged roots at appropriate positions.
    iterations : ndarray
        Original full-size array of iteration counts (int64). Contains iteration
        counts from previous attempts. Modified in-place to update counts for
        newly converged elements.
    converged_flag : ndarray
        Original full-size boolean array indicating convergence status. False for
        unconverged elements, True for converged. Modified in-place to set True
        for newly converged positions.
    unconverged_idx : ndarray of int
        Integer indices mapping from the subset (updated_*) to the full arrays.
        These are the positions in the original arrays that were unconverged
        before this solver attempt.

        Example: If unconverged_idx = [1, 3, 4], then:
        - updated_roots[0] corresponds to roots[1]
        - updated_roots[1] corresponds to roots[3]
        - updated_roots[2] corresponds to roots[4]
    updated_roots : ndarray
        Subset of newly computed root values from the backup solver (float64).
        Length equals len(unconverged_idx). Values at positions where
        updated_converged_flag is True will be copied to the original roots array.
    updated_iterations : ndarray
        Subset of newly computed iteration counts from the backup solver (int64).
        Length equals len(unconverged_idx). Values for converged elements will be
        copied to the original iterations array.
    updated_converged_flag : ndarray
        Boolean array indicating which elements in the subset converged (bool).
        Length equals len(unconverged_idx). True means the solver succeeded for
        that element; False means it failed and should be left for the next backup.

    Returns
    -------
    None
        This function modifies the input arrays in-place and returns nothing.
        After execution, roots, iterations, and converged_flag will contain
        updated values for newly converged elements.

    See Also
    --------
    _get_unconverged_func_params : Counterpart for extracting subsets
    _try_back_up_open_vectorised : Uses this to update after open methods
    _try_back_up_bracket_vectorised : Uses this to update after bracket methods
    """
    updated_converged_mask = updated_converged_flag
    updated_converged_original_idx = unconverged_idx[updated_converged_mask]

    roots[updated_converged_original_idx] = updated_roots[updated_converged_mask]
    iterations[updated_converged_original_idx] = updated_iterations[updated_converged_mask]
    converged_flag[updated_converged_original_idx] = True
