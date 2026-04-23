"""
Jit equation for bolton lcl approximation equations.

Author: Cian Quezon
"""
import numpy as np
import numpy.typing as npt
from numba import njit, prange

from meteocalc.shared.constants import Rd, cpd


@njit
def _bolton_lcl_temp_scalar(temp_k: float, dewpoint_temp_k: float):
    """
    Compute the LCL temperature for a single parcel using Bolton (1980).

    Returns the temperature at which an air parcel becomes saturated when
    lifted dry-adiabatically from the surface. Implements Bolton (1980)
    eq. 15, accurate to within ~1 K for typical atmospheric conditions.

    Parameters
    ----------
    temp_k : float
        Surface air temperature in Kelvin.
    dewpoint_temp_k : float
        Surface dewpoint temperature in Kelvin.

    Returns
    -------
    float
        Temperature at the LCL in Kelvin.
    """
    return 56.0 + 1.0 / (
        1.0 / (dewpoint_temp_k - 56.0) + np.log(temp_k / dewpoint_temp_k) / 800.0
    )


@njit
def _bolton_lcl_pressure_scalar(temp_k: float, lcl_temp_k: float, pressure_hpa: float):
    """
    Compute the LCL pressure for a single parcel via Poisson's relation.

    Given the LCL temperature, derives the corresponding pressure level
    using the dry adiabatic Poisson relation.

    Parameters
    ----------
    temp_k : float
        Surface air temperature in Kelvin.
    lcl_temp_k : float
        Temperature at the LCL in Kelvin, as returned by
        ``_bolton_lcl_temp_scalar``.
    pressure_hpa : float
        Surface pressure in hectopascals (hPa).

    Returns
    -------
    float
        Pressure at the LCL in hectopascals (hPa).
    """
    return pressure_hpa * (lcl_temp_k / temp_k) ** (cpd / Rd)


@njit
def _bolton_lcl_scalar(temp_k: float, dewpoint_temp_k: float, pressure_hpa: float):
    """
    Compute LCL temperature and pressure for a single parcel.

    Primary scalar interface for LCL calculations. Combines
    ``_bolton_lcl_temp_scalar`` and ``_bolton_lcl_pressure_scalar``,
    computing the LCL temperature once and reusing it for the pressure
    calculation.

    Parameters
    ----------
    temp_k : float
        Surface air temperature in Kelvin.
    dewpoint_temp_k : float
        Surface dewpoint temperature in Kelvin.
    pressure_hpa : float
        Surface pressure in hectopascals (hPa).

    Returns
    -------
    lcl_temp_k : float
        Temperature at the LCL in Kelvin.
    lcl_pressure_hpa : float
        Pressure at the LCL in hectopascals (hPa).
    """
    lcl_temp_k = _bolton_lcl_temp_scalar(temp_k=temp_k, dewpoint_temp_k=dewpoint_temp_k)
    lcl_pressure_hpa = _bolton_lcl_pressure_scalar(
        temp_k=temp_k, lcl_temp_k=lcl_temp_k, pressure_hpa=pressure_hpa
    )

    return lcl_temp_k, lcl_pressure_hpa


@njit(parallel=True)
def _bolton_lcl_vectorised(
    temp_k: npt.ArrayLike, dewpoint_temp_k: npt.ArrayLike, pressure_hpa: npt.ArrayLike
):
    """
    Compute LCL temperature and pressure for an array of parcels.

    Vectorised equivalent of ``_bolton_lcl_scalar``, parallelised across
    CPU cores via ``numba.prange``.

    Parameters
    ----------
    temp_k : array-like of float
        1-D array of surface air temperatures in Kelvin.
    dewpoint_temp_k : array-like of float
        1-D array of surface dewpoint temperatures in Kelvin.
        Must be the same length as ``temp_k``.
    pressure_hpa : array-like of float
        1-D array of surface pressures in hectopascals (hPa).
        Must be the same length as ``temp_k``.

    Returns
    -------
    lcl_temp_k : np.ndarray of float64
        LCL temperatures in Kelvin, shape ``(n,)``.
    lcl_pressure_hpa : np.ndarray of float64
        LCL pressures in hectopascals (hPa), shape ``(n,)``.
    """

    n = len(temp_k)
    results_lcl_temp = np.empty(n, dtype=np.float64)
    results_lcl_press = np.empty(n, dtype=np.float64)

    for i in prange(n):
        results_lcl_temp[i], results_lcl_press[i] = _bolton_lcl_scalar(
            temp_k=temp_k[i],
            dewpoint_temp_k=dewpoint_temp_k[i],
            pressure_hpa=pressure_hpa[i],
        )

    return results_lcl_temp, results_lcl_press
