"""
Jit equations for thermodynamics.

Author: Cian Quezon
"""
import numpy.typing as npt
import numpy as np
import math
from numba import njit, prange


@njit 
def _bolton_theta_e_scalar(temp_k: float, lcl_temp_k: float,
                                      pressure_hpa: float, vapor_pressure_hpa: float,
                                      mixing_ratio: float, p0: float = 1000.0):
    """
    Calculate equivalent potential temperature (θE) using Bolton (1980) method.
    
    Combines potential temperature at LCL (θL) with exponential moisture term
    to represent total thermal and latent heat energy content of an air parcel.
    Uses Bolton equation 39 for the exponential moisture correction.
    
    Parameters
    ----------
    temp_k : float
        Air temperature in Kelvin
    lcl_temp_k : float
        Lifting condensation level temperature in Kelvin
    pressure_hpa : float
        Total atmospheric pressure in hPa
    vapor_pressure_hpa : float
        Water vapor pressure in hPa
    mixing_ratio : float
        Water vapor mixing ratio in kg/kg
    p0 : float, optional
        Reference pressure in hPa, default 1000.0
        
    Returns
    -------
    float
        Equivalent potential temperature in Kelvin
        
    """

    theta_L = _bolton_theta_l_scalar(temp_k=temp_k, lcl_temp_k=lcl_temp_k, pressure_hpa=pressure_hpa,
                                         vapor_pressure_hpa=vapor_pressure_hpa, mixing_ratio=mixing_ratio, p0=p0)
    
    exponential_term = ((3376.0/lcl_temp_k) - 0.00254) * mixing_ratio * (1.0 + 0.81 * mixing_ratio)

    if exponential_term > 50.0:
        return math.nan
    
    theta_E = theta_L * math.exp(exponential_term)

    return theta_E

@njit
def _bolton_theta_e_vectorised(temp_k: npt.ArrayLike, lcl_temp_k: npt.ArrayLike,
                                      pressure_hpa: npt.ArrayLike, vapor_pressure_hpa: npt.ArrayLike,
                                      mixing_ratio: npt.ArrayLike, p0: npt.ArrayLike):
    """
    Vectorized calculation of equivalent potential temperature (θE).
    
    Applies scalar equivalent potential temperature calculation to input arrays
    using parallel processing for high-performance computation of large datasets.
    
    Parameters
    ----------
    temp_k : npt.ArrayLike
        Air temperature array in Kelvin
    lcl_temp_k : npt.ArrayLike
        Lifting condensation level temperature array in Kelvin
    pressure_hpa : npt.ArrayLike
        Total atmospheric pressure array in hPa
    vapor_pressure_hpa : npt.ArrayLike
        Water vapor pressure array in hPa
    mixing_ratio : npt.ArrayLike
        Water vapor mixing ratio array in kg/kg
    p0 : npt.ArrayLike
        Reference pressure in hPa, default 1000.0
        
    Returns
    -------
    np.ndarray
        Equivalent potential temperature array in Kelvin
    """
    n = len(temp_k)
    results = np.empty(n, dtype=np.float64)

    for i in prange(n):
        results[i] = _bolton_theta_e_scalar(temp_k=temp_k[i], lcl_temp_k=lcl_temp_k[i],
                                      pressure_hpa=pressure_hpa[i], vapor_pressure_hpa=vapor_pressure_hpa[i],
                                      mixing_ratio=mixing_ratio[i], p0=p0[i])
    
    return results

@njit
def _bolton_theta_l_scalar(temp_k: float, lcl_temp_k: float,
                               pressure_hpa: float, vapor_pressure_hpa: float,
                               mixing_ratio: float, p0: float = 1000.0):
    """
    Calculate potential temperature at lifting condensation level (θL).
    
    Implements Bolton (1980) equation 24 with moisture corrections for
    dry air pressure and thermodynamic effects of water vapor.
    
    Parameters
    ----------
    temp_k : float
        Air temperature in Kelvin
    lcl_temp_k : float
        Lifting condensation level temperature in Kelvin
    pressure_hpa : float
        Total atmospheric pressure in hPa
    vapor_pressure_hpa : float
        Water vapor pressure in hPa
    mixing_ratio : float
        Water vapor mixing ratio in kg/kg
    p0 : float, optional
        Reference pressure in hPa, default 1000.0
        
    Returns
    -------
    float
        Potential temperature at LCL in Kelvin
    """
    Rd = 287.04
    cp = 1004.0
    kappa = Rd/cp

    pressure_term = (p0 / (pressure_hpa - vapor_pressure_hpa)) ** kappa

    moisture_term = (temp_k/lcl_temp_k) ** (0.28 * mixing_ratio)

    return temp_k * pressure_term * moisture_term

@njit
def _bolton_theta_l_vectorised(temp_k: npt.ArrayLike, lcl_temp_k: npt.ArrayLike,
                               pressure_hpa: npt.ArrayLike, vapor_pressure_hpa: npt.ArrayLike,
                               mixing_ratio: npt.ArrayLike, p0: npt.ArrayLike):
    """
    Vectorized calculation of potential temperature at LCL (θL).
    
    Applies scalar LCL potential temperature calculation to input arrays
    using parallel processing for high-performance computation.
    
    Parameters
    ----------
    temp_k : npt.ArrayLike
        Air temperature array in Kelvin
    lcl_temp_k : npt.ArrayLike 
        LCL temperature array in Kelvin
    pressure_hpa : npt.ArrayLike
        Total atmospheric pressure array in hPa
    vapor_pressure_hpa : npt.ArrayLike
        Water vapor pressure array in hPa
    mixing_ratio : npt.ArrayLike
        Water vapor mixing ratio array in kg/kg
    p0 : npt.ArrayLike
        Reference pressure in hPa, default 1000.0
        
    Returns
    -------
    np.ndarray
        Potential temperature at LCL array in Kelvin
        
    """
    n = len(temp_k)
    results = np.empty(n, dtype=np.float64)

    for i in prange(n):
        results[i] = _bolton_theta_l_scalar(temp_k=temp_k[i], lcl_temp_k=lcl_temp_k[i],
                                                pressure_hpa=pressure_hpa[i], vapor_pressure_hpa=vapor_pressure_hpa[i],
                                                 mixing_ratio=mixing_ratio[i], p0=p0[i])
    
    return results


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