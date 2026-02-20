"""
Jit equation for approximation equations.

Author: Cian Quezon
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit
def _magnus_equation_scalar(temp_k: float, rh: float, A: float, B: float) -> float:
    """
    Scalar Magnus-Tetens formula for dew point calculation.

    Calculates dew point temperature for a single temperature-humidity pair
    using the Magnus-Tetens approximation formula. JIT-compiled with Numba
    for high performance.

    Parameters
    ----------
    temp_k : float
        Air temperature in Kelvin
    rh : float
        Relative humidity as fraction (0-1, not percentage)
    A : float
        Magnus equation coefficient A
    B : float
        Magnus equation coefficient B in Celsius

    Returns
    -------
    float
        Dew/frost point temperature in Kelvin

    See Also
    --------
    _magnus_equation_vectorised : Parallel version for arrays
    MagnusDewpointEquation : High-level interface
    """

    temp_c = temp_k - 273.15

    alpha = ((A * temp_c) / (B + temp_c)) + np.log(rh)
    td_c = (B * alpha) / (A - alpha)

    return td_c + 273.15


@njit(parallel=True, fastmath=True)
def _magnus_equation_vectorised(
    temp_k: npt.ArrayLike, rh: npt.ArrayLike, A: float, B: float
) -> npt.NDArray:
    """
    Vectorized Magnus formula for dew point calculation with parallel processing.

    Calculates dew point temperatures for arrays of temperature and humidity
    values using the Magnus-Tetens approximation formula. Optimized with
    Numba JIT compilation and parallel processing via prange.

    Parameters
    ----------
    temp_k : ndarray
        Air temperature(s) in Kelvin, shape (n,)
    rh : ndarray
        Relative humidity(ies) as fraction (0-1), shape (n,)
    A : float
        Magnus equation coefficient A
    B : float
        Magnus equation coefficient B (Celsius)

    Returns
    -------
    ndarray
        Dew/frost point temperature(s) in Kelvin, shape (n,)

    See Also
    --------
    _magnus_equation_scalar : Scalar version for single calculations
    MagnusDewpointEquation : High-level interface
    """

    n = len(temp_k)
    results = np.empty(n, dtype=np.float64)

    for i in prange(n):
        results[i] = _magnus_equation_scalar(temp_k=temp_k[i], rh=rh[i], A=A, B=B)
    return results
