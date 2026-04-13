"""
Jit equations for thermodynamics.

Author: Cian Quezon
"""
import numpy.typing as npt
import numpy as np

from numba import njit, prange


@njit
def _poisson_scalar(temp_k: float, p: float, p0: float = 1000.0):
    """
    Compute potential temperature using Poisson's equation.
    
    Parameters
    ----------
    temp_k  : temperature (K)
    p  : pressure (hPa)
    p0 : reference pressure (hPa), default 1000 hPa
    
    Returns
    -------
    theta : potential temperature (K)
    """
    Rd = 287.04
    cp = 1004.0
    kappa = Rd/cp

    return temp_k * (p0 / p) ** kappa

@njit
def _poisson_vectorised(temp_k: npt.ArrayLike, p: npt.ArrayLike, p0: npt.ArrayLike):
    """
    Compute potential temperature using Poisson's equation.
    
    Parameters
    ----------
    temp_k  : temperature array (K)
    p  : pressure array (hPa)
    p0 : reference pressure array (hPa), default 1000 hPa
    
    Returns
    -------
    theta : potential temperature array (K)
    """
    n = len(temp_k)
    results = np.empty(n, dtype=np.float64)

    for i in prange(n):
        results[i] = _poisson_scalar(temp_k=temp_k[i], p=p[i], p0=p0)
    
    return results