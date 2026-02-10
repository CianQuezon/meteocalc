"""
Docstring for meteocalc.dewpoint._jit_equations
"""
import numpy as np
import numpy.typing as npt

from numba import njit, prange

@njit
def _magnus_equation_scalar(temp_k: float, rh: float, 
                            A: float, B: float) -> float:
    """
    Docstring for _magnus_equation_scalar
    
    :param temp_k: Description
    :type temp_k: float
    :param rh: Description
    :type rh: float
    :param A: Description
    :type A: float
    :param B: Description
    :type B: float
    :return: Description
    :rtype: float
    """

    temp_c = temp_k - 273.15

    alpha = ((A * temp_c) / (B + temp_c)) + np.log(rh)
    td_c = (B * alpha) / (A - alpha)

    return td_c + 273.15

@njit(parallel=True, fastmath=True)
def _magnus_equation_vectorised(temp_k: npt.ArrayLike, rh: npt.ArrayLike,
                                A: float, B: float) -> npt.NDArray:
    """
    Docstring for _magnus_equation_vectorised
    
    :param temp_k: Description
    :type temp_k: npt.ArrayLike
    :param rh: Description
    :type rh: npt.ArrayLike
    :param A: Description
    :type A: float
    :param B: Description
    :type B: float
    :return: Description
    :rtype: NDArray
    """

    n = len(temp_k)
    results = np.empty(n, dtype=np.float64)

    for i in prange(n):
        results[i] = _magnus_equation_scalar(temp_k=temp_k[i], rh=rh[i], A=A, B=B)
    return results