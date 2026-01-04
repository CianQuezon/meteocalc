"""
Root finding solvers with using Numba JIT compilation.

Author: Cian Quezon
"""

from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit
def _newton_raphson_scalar(
    func: Callable[[float], float],
    func_prime: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[float, int, bool]:
    """
    Newton raphson for root finding.

    Args:
        func(Callable[[float], float]) = Function of the root required to solve
        func_prime(Callable[[float], float]) = Derivative of the function
        x0(float) = Initial guess
        tol(float) = Tolerance for convergence
        max_iter(int) = Maximum iterations

    Returns:
        (root, iterations, converged)
    """
    x = x0

    for i in range(max_iter):
        fx = func(x)
        fpx = func_prime(x)

        if abs(fpx) < 1e-15:
            return x, i, False

        x_new = x - (fx / fpx)

        if abs(x_new - x) < tol:
            return x_new, i + 1, True

        x = x_new

    return x, max_iter, False


@njit(parallel=True)
def _newton_raphson_vectorised(
    func: Callable[[float], float],
    func_prime: Callable[[float], float],
    x0: npt.ArrayLike,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """
    Vectorised version for Newton Raphson solver

    Args:
        - func(Callable[[float], float]) = function required to solve the root
        - func_prime(Callable[[float], float]) = derivative of the function
        - x0(npt.ArrayLike) = Array of initial guesses
        - tol (float) = Convergence tolerance
        - max_iter(int) = maximum amount of iterations

    Returns:
        Array of (root, iterations, converged)
    """
    n = len(x0)
    root_arr = np.empty(n, dtype=np.float64)
    iterations_arr = np.empty(n, dtype=np.int64)
    converged_arr = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        root, iteration, converged = _newton_raphson_scalar(
            func=func, func_prime=func_prime, x0=x0[i], tol=tol, max_iter=max_iter
        )
        root_arr[i] = root
        iterations_arr[i] = iteration
        converged_arr[i] = converged

    return root_arr, iterations_arr, converged_arr


@njit
def _bisection_scalar(
    func: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100
) -> Tuple[float, int, bool]:
    """
    Scalar bisection to find roots.

    Args:
        - func(Callable[[float], float]) = Function of the root required to solve
        - a (float) = Lower bracket bound
        - b (float) = Upper bracket bound
        - tol (float) = Tolerance for convergence
        - max_iter(int) = Maximum iterations

    Returns:
        - (root, iterations, converged)

    """
    fa = func(a)
    fb = func(b)

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
        fc = func(c)

        if abs(fc) < tol or abs(b - a) / 2.0 < tol:
            return c, i + 1, True
        if fc < 0.0:
            a = c
            fa = fc
        else:
            b = c
            fb = fc

    return (a + b) / 2.0, max_iter, False


@njit(parallel=True)
def _bisection_vectorised(
    func: Callable[[float], float],
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """
    Vectorised version bisection method for arrays.

    Args:
        - func: Callable[[float], float] = Function to solve for the root
        - a (npt.NDArray) = Upper bracket bound
        - b (npt.NDArray) = Lower bracket bound
        - tol (float) = Convergence tolerance
        - max_iter (int) = Maximum amount of iterations

    Returns:
        - Array of roots in (root, iterations, converged)
    """

    n = len(a)
    root_arr = np.empty(n, dtype=np.float64)
    iterations_arr = np.empty(n, dtype=np.int64)
    converged_arr = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        root, iteration, converged = _bisection_scalar(
            func=func, a=a[i], b=b[i], tol=tol, max_iter=max_iter
        )
        root_arr[i] = root
        iterations_arr[i] = iteration
        converged_arr[i] = converged

    return root_arr, iterations_arr, converged_arr


@njit
def _brent_scalar(
    func: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100
) -> Tuple[float, int, bool]:
    """
    Brent's method to find roots.

    Args:
        - func(Callable[[float], float]) Function of the root required to be solved
        - a (float) = Lower bracket bound
        - b (float) = Upper bracket bound
        - tol (float) = Tolerance for convergence
        - max_iter (int) = maximum iterations

    Returns:
        - (root, iterations, converged)

    """
    fa = func(a)
    fb = func(b)

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

        fs = func(s)

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


@njit(parallel=True)
def _brent_vectorised(
    func: Callable[[float], float],
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """
    Vectorised version brent method for arrays.

    Args:
        - func: Callable[[float], float] = Function to solve for the root
        - a (npt.NDArray) = Upper bracket bound
        - b (npt.NDArray) = Lower bracket bound
        - tol (float) = Convergence tolerance
        - max_iter (int) = Maximum amount of iterations

    Returns:
        - Array of roots in (root, iterations, converged)
    """

    n = len(a)
    root_arr = np.empty(n, dtype=np.float64)
    iterations_arr = np.empty(n, dtype=np.int64)
    converged_arr = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        root, iteration, converged = _brent_scalar(
            func=func, a=a[i], b=b[i], tol=tol, max_iter=max_iter
        )
        root_arr[i] = root
        iterations_arr[i] = iteration
        converged_arr[i] = converged
    return root_arr, iterations_arr, converged_arr
