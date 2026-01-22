"""
Root finding solvers with using Numba JIT compilation.

Author: Cian Quezon
"""

from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from meteorological_equations.math.solvers._codegen import generate_vectorised_solver
from meteorological_equations.math.solvers._enums import MethodType


@njit
def _newton_raphson_scalar(
    func: Callable[[float], float],
    func_prime: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 50,
    *func_params,
) -> Tuple[float, int, bool]:
    """
    Scalar Newton-Raphson method for finding roots of a function.
    
    This is the core JIT-compiled implementation of Newton's method for a
    single root-finding problem. It uses the tangent line approximation
    to iteratively refine an initial guess until convergence or failure.
    
    The method computes: x_{n+1} = x_n - f(x_n) / f'(x_n)
    
    This function is designed to be called by vectorised solvers and should
    generally not be called directly by users.
    
    Parameters
    ----------
    func : callable
        Function for which to find the root. Must be JIT-compiled and have
        signature: func(x, *params) -> float, where x is the independent
        variable and params are optional additional parameters.
    func_prime : callable
        Derivative of func. Must be JIT-compiled and have the same signature
        as func: func_prime(x, *params) -> float.
    x0 : float
        Initial guess for the root location. The quality of this guess
        significantly affects convergence. Should be reasonably close to
        the actual root for best results.
    tol : float, default=1e-6
        Convergence tolerance. The algorithm stops when the absolute change
        in x between iterations is less than tol: |x_{n+1} - x_n| < tol.
    max_iter : int, default=50
        Maximum number of iterations allowed. If convergence is not achieved
        within this limit, the function returns with converged=False.
    *func_params : tuple
        Variable-length tuple of additional parameters to pass to func and
        func_prime. These are unpacked and passed as: func(x, *func_params).
    
    Returns
    -------
    root : float
        Estimated root location. If converged=True, this satisfies the
        convergence criterion. If converged=False, this is the best estimate
        after max_iter iterations or when derivative becomes too small.
    iterations : int
        Number of iterations performed. Range: [0, max_iter].
        - If converged=True: actual iterations to convergence
        - If converged=False: max_iter (or iteration where derivative failed)
    converged : bool
        Convergence flag indicating whether the solution met the tolerance
        criterion within max_iter iterations.
        - True: |x_{n+1} - x_n| < tol (successful convergence)
        - False: max_iter reached or derivative too small (|f'(x)| < 1e-15)
    
    See Also
    --------
    _newton_raphson_vectorised : Vectorised version for solving multiple problems
    _bisection_scalar : Bracket-based alternative (no derivative needed)
    _brent_scalar : Robust bracket-based method (recommended for most cases)
    """
    x = x0

    for i in range(max_iter):
        fx = func(x, *func_params)
        fpx = func_prime(x, *func_params)

        if abs(fpx) < 1e-15:
            return x, i, False

        x_new = x - (fx / fpx)

        if abs(x_new - x) < tol:
            return x_new, i + 1, True

        x = x_new

    return x, max_iter, False


def _newton_raphson_vectorised(
    func: Callable[[float], float],
    func_prime: Callable[[float], float],
    x0: npt.ArrayLike,
    func_params: Optional[npt.ArrayLike] = None,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """
    Vectorised Newton-Raphson method for solving multiple root-finding problems.
    
    This function solves multiple independent root-finding problems in parallel
    using Newton's method. Each problem can have different initial guesses and
    different function parameters, enabling efficient batch solving of related
    problems (e.g., finding equilibrium conditions at multiple meteorological
    stations).
    
    The vectorisation is achieved through Numba's prange for automatic
    parallelization across CPU cores, with runtime code generation handling
    variable numbers of function parameters.
    
    Parameters
    ----------
    func : callable
        Function for which to find roots. Must be JIT-compiled (@njit) and
        have signature: func(x, *params) -> float, where x is the independent
        variable and params are optional additional parameters.
        
        The same function is used for all problems, but parameters can vary
        per problem via func_params.
    func_prime : callable
        Derivative of func with respect to x. Must be JIT-compiled (@njit)
        and have the same signature as func: func_prime(x, *params) -> float.
        
        Accurate derivatives are critical for Newton's method convergence.
    x0 : array_like, shape (n_solves,)
        Array of initial guesses, one per problem. Each problem uses its
        corresponding initial guess for Newton iteration.
        
        Quality of initial guesses significantly affects convergence rate
        and reliability. Poor guesses may cause divergence or slow convergence.
    func_params : array_like or None, optional
        Function parameters for each problem. Can be:
        
        - None: No parameters (func and func_prime take only x)
        - 1D array, shape (n_params,): Same parameters for ALL problems
          (broadcast to all)
        - 2D array, shape (n_solves, n_params): Different parameters per problem
        
        Default is None.
    tol : float, default=1e-6
        Convergence tolerance for all problems. Each problem stops when
        |x_{n+1} - x_n| < tol.
    max_iter : int, default=50
        Maximum iterations allowed per problem. Problems that don't converge
        within this limit return with converged=False.
    
    Returns
    -------
    roots : ndarray, shape (n_solves,), dtype=float64
        Array of root locations, one per problem. For converged problems,
        these satisfy the tolerance criterion. For unconverged problems,
        these are the best estimates after max_iter iterations.
    iterations : ndarray, shape (n_solves,), dtype=int64
        Number of iterations performed for each problem. Range: [0, max_iter].
        Lower values indicate faster convergence (or early failure).
    converged : ndarray, shape (n_solves,), dtype=bool
        Convergence flags for each problem:
        
        - True: Problem converged (|x_{n+1} - x_n| < tol)
        - False: Problem failed (max_iter reached or derivative too small)
    
    See Also
    --------
    _newton_raphson_scalar : Scalar version for single problem
    _bisection_vectorised : Vectorised bracket method (no derivative needed)
    _brent_vectorised : Vectorised robust bracket method (recommended alternative)
    generate_vectorised_solver : Code generator used internally
    _validate_and_prepare_params : Parameter validation and preparation
    
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n_solves = len(x0)

    func_params, num_params = _validate_and_prepare_params(
        func_params=func_params, n_solves=n_solves
    )

    solver = generate_vectorised_solver(
        scalar_func=_newton_raphson_scalar, num_params=num_params, method_type=MethodType.OPEN
    )

    return solver(func, func_prime, func_params, x0, tol, max_iter)


@njit
def _bisection_scalar(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    *func_params,
) -> Tuple[float, int, bool]:
    """
    Scalar bisection to find roots.

    Args:
        - func(Callable[[float], float]) = Function of the root required to solve
        - a (float) = Lower bracket bound
        - b (float) = Upper bracket bound
        - tol (float) = Tolerance for convergence
        - max_iter(int) = Maximum iterations
        - func_params = function parameters

    Returns:
        - (root, iterations, converged)

    """
    fa = func(a, *func_params)
    fb = func(b, *func_params)

    if fa == 0.0:
        return a, 0, True
    if fb == 0.0:
        return b, 0, True

    if fa * fb > 0.0:
        return np.nan, 0, False

    if fa > 0.0:
        a, b = b, a
        fa, fb = fb, fa

    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = func(c, *func_params)

        if abs(fc) < tol or abs(b - a) / 2.0 < tol:
            return c, i + 1, True
        if fc < 0.0:
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    return (a + b) / 2.0, max_iter, False


def _bisection_vectorised(
    func: Callable[[float], float],
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    func_params: Optional[npt.ArrayLike] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """
    Vectorised version bisection method for arrays.

    Args:
        - func (Callable[[float], float]) = Function to solve for the root
        - func_params (npt.ArrayLike) = An array of function parameters
        - a (npt.NDArray) = Upper bracket bound
        - b (npt.NDArray) = Lower bracket bound
        - tol (float) = Convergence tolerance
        - max_iter (int) = Maximum amount of iterations

    Returns:
        - Array of roots in (root, iterations, converged)
    """
    a = np.asarray(a, dtype=np.float64)
    n_solves = len(a)

    func_params, num_params = _validate_and_prepare_params(
        func_params=func_params, n_solves=n_solves
    )

    solver = generate_vectorised_solver(
        scalar_func=_bisection_scalar, num_params=num_params, method_type=MethodType.BRACKET
    )

    return solver(func, func_params, a, b, tol, max_iter)


@njit
def _brent_scalar(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    *func_params,
) -> Tuple[float, int, bool]:
    """
    Brent's method to find roots.

    Args:
        - func(Callable[[float], float]) Function of the root required to be solved
        - a (float) = Lower bracket bound
        - b (float) = Upper bracket bound
        - tol (float) = Tolerance for convergence
        - max_iter (int) = maximum iterations
        - func_params = function parameters for the function

    Returns:
        - (root, iterations, converged)

    """
    fa = func(a, *func_params)
    fb = func(b, *func_params)

    if fa == 0.0:
        return a, 0, True
    if fb == 0.0:
        return b, 0, True

    if fa * fb > 0.0:
        return np.nan, 0, False

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = 0.0
    mflag = True

    for i in range(max_iter):
        if abs(fb) < tol or abs(b - a) < tol:
            return b, i + 1, True

        if fa != fc and fb != fc:
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )

        else:
            s = b - fb * (b - a) / (fb - fa)

        tmp2 = (3 * a + b) / 4.0

        condition1 = not ((s > tmp2 and s < b) or (s < tmp2 and s > b))
        condition2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        condition3 = not mflag and abs(s - b) >= abs(c - d) / 2.0
        condition4 = mflag and abs(b - c) < tol
        condition5 = not mflag and abs(c - d) < tol

        if condition1 or condition2 or condition3 or condition4 or condition5:
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        fs = func(s, *func_params)

        d = c
        c = b
        fc = fb

        if fa * fs < 0.0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b, max_iter, False


def _brent_vectorised(
    func: Callable[[float], float],
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    func_params: Optional[npt.ArrayLike] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """
    Vectorised version brent method for arrays.

    Args:
        - func: Callable[[float], float] = Function to solve for the root
        - func_params (npt.ArrayLike) = An array of function parameters
        - a (npt.NDArray) = Upper bracket bound
        - b (npt.NDArray) = Lower bracket bound
        - tol (float) = Convergence tolerance
        - max_iter (int) = Maximum amount of iterations

    Returns:
        - Array of roots in (root, iterations, converged)
    """

    a = np.asarray(a, dtype=np.float64)
    n_solves = len(a)

    func_params, num_params = _validate_and_prepare_params(
        func_params=func_params, n_solves=n_solves
    )

    solver = generate_vectorised_solver(
        scalar_func=_brent_scalar, num_params=num_params, method_type=MethodType.BRACKET
    )

    return solver(func, func_params, a, b, tol, max_iter)


def _validate_and_prepare_params(
    func_params: Optional[npt.ArrayLike],
    n_solves: int,
) -> Tuple[npt.NDArray[np.float64], int]:
    """
    Validates and prepares the function parameters for vectorised solvers

    Args:
        - func_params (npt.ArrayLike) = Array of function parameters or None for no parameters
        - n_solves (int) = Number of solves

    Returns:
        - (prepared_params, num_params) = Prepared function parameters and number of parameters
    """
    if func_params is None:
        return np.empty((n_solves, 0), dtype=np.float64), 0

    func_params = np.asarray(func_params, dtype=np.float64)

    if func_params.ndim == 1:
        if len(func_params) != n_solves:
            raise ValueError(
                f"func_params length ({len(func_params)}) must match number of solves ({n_solves})"
            )
        num_params = 1
        func_params = func_params.reshape(-1, 1)

    else:
        if func_params.shape[0] != n_solves:
            raise ValueError(
                f"func_params rows ({func_params.shape[0]}) must matchnumber of solves ({n_solves})"
            )

        num_params = func_params.shape[1]

    return func_params, num_params
