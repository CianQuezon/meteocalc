"""
Docstring for meteorological_equations.meteorological_equations.vapor.__jit_funcs

This code is for calculating the equations for different vapor saturation equations.
It uses numba for optimal speed.


Equation:
- Goff & Gratch
- Bolton

Author: Cian Quezon
"""

import numpy as np
import numpy.typing as npt

from numba.typed import Dict as TypedDict
from numba import njit, prange, types
from typing import Dict


@njit
def _bolton_scalar(temp_c: float) -> float:
    """
    Calculate saturation vapor pressure using Bolton's equation.

    Args:
        temp_c: Temperature in degrees (Celsius)
    
    Returns:
        Saturation vapor pressure in hPa (millibars)
    """
    return 6.112 * np.exp((17.67 * temp_c)/(temp_c + 243.5))

@njit(parallel=True, fastmath=True)
def _bolton_vectorised(temp_c: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Docstring for _bolton_vectorised
    
    :param temp_c: Description
    :type temp_c: npt.ArrayLike
    :return: Description
    :rtype: NDArray[float64]
    
    """

    n = len(temp_c)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        result[i] = _bolton_scalar(temp_c[i])
    return result


