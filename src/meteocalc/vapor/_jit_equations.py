"""
Calculates different vapor saturations with different equations using Numba.

Equation:
- Goff & Gratch
- Bolton
- Hyland Wexler

Author: Cian Quezon
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit
def _bolton_scalar(temp_k: float) -> float:
    """Calculate saturation vapor pressure using Bolton's equation.

    Parameters
    ----------
    temp_k : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Saturation vapor pressure in hPa.
    """
    temp_c = temp_k - 273.15
    return float(6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5)))


@njit(parallel=True, fastmath=True)
def _bolton_vectorised(temp_k: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Calculate saturation vapor pressure arrays using Bolton's equation.

    Parameters
    ----------
    temp_k : array_like
        Array of temperatures in Kelvin.

    Returns
    -------
    ndarray
        Saturation vapor pressure in hPa.
    """

    n = len(temp_k)
    result = np.empty(n, dtype=np.float64)

    for i in prange(n):
        result[i] = _bolton_scalar(temp_k[i])
    return result


@njit
def _goff_gratch_scalar(
    temp_k: float,
    T_ref: float,
    A: float,
    B: float,
    C: float,
    C_exp: float,
    D: float,
    D_exp: float,
    log_p_ref: float,
) -> float:
    """Calculate saturation vapor pressure using Goff-Gratch equation.

    Implements the Goff-Gratch equation. Ice equation omits D term (D=0, D_exp=0).

    Parameters
    ----------
    temp_k : float
        Temperature in Kelvin.
    T_ref : float
        Reference temperature in Kelvin.
    A, B, C, D : float
        Equation coefficients.
    C_exp, D_exp : float
        Exponent coefficients.
    log_p_ref : float
        Log10 of reference pressure.

    Returns
    -------
    float
        Saturation vapor pressure in hPa.
    """

    A_sum = A * (T_ref / temp_k - 1)
    B_sum = B * np.log10(T_ref / temp_k)

    if C_exp == 0.0:
        C_sum = C * (1 - temp_k / T_ref)

    else:
        C_sum = C * (10 ** (C_exp * (1 - temp_k / T_ref)) - 1)

    D_sum = 0.0 if D == 0.0 else D * (10 ** (D_exp * (T_ref / temp_k - 1)) - 1)

    log_ew = A_sum + B_sum + C_sum + D_sum + log_p_ref
    return float(10**log_ew)


@njit(parallel=True, fastmath=True)
def _goff_gratch_vectorised(
    temp_k: npt.ArrayLike,
    T_ref: float,
    A: float,
    B: float,
    C: float,
    C_exp: float,
    D: float,
    D_exp: float,
    log_p_ref: float,
) -> npt.NDArray[np.float64]:
    """Calculate saturation vapor pressure arrays using Goff-Gratch equation.

    Vectorized implementation. Ice equation omits D term (D=0, D_exp=0).

    Parameters
    ----------
    temp_k : array_like
        Array of temperatures in Kelvin.
    T_ref : float
        Reference temperature in Kelvin.
    A, B, C, D : float
        Equation coefficients.
    C_exp, D_exp : float
        Exponent coefficients.
    log_p_ref : float
        Log10 of reference pressure.

    Returns
    -------
    ndarray
        Saturation vapor pressure in hPa.
    """
    n = len(temp_k)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _goff_gratch_scalar(
            temp_k[i], T_ref, A, B, C, C_exp, D, D_exp, log_p_ref
        )
    return result


@njit
def _hyland_wexler_scalar(
    temp_k: float, A: float, B: float, C: float, D: float, E: float, F: float, G: float
) -> float:
    """Calculate saturation vapor pressure arrays using Hyland-Wexler equation.

    Parameters
    ----------
    temp_k : array_like
        Array of temperatures in Kelvin.
    A : float
        Coefficient for 1/T term [K].
    B : float
        Constant coefficient [dimensionless].
    C : float
        Coefficient for T term [K⁻¹].
    D : float
        Coefficient for T² term [K⁻²].
    E : float
        Coefficient for T³ term [K⁻³].
    F : float
        Coefficient for T⁴ term [K⁻⁴].
    G : float
        Coefficient for ln(T) term [dimensionless].

    Returns
    -------
    ndarray
        Saturation vapor pressure in hPa.
    """
    A_sum = A / temp_k
    B_sum = B
    C_sum = C * temp_k
    D_sum = D * (temp_k**2)
    E_sum = E * (temp_k**3)
    F_sum = F * (temp_k**4)
    G_sum = G * np.log(temp_k)

    ln_ew = A_sum + B_sum + C_sum + D_sum + E_sum + F_sum + G_sum

    return float(np.exp(ln_ew) / 100)


@njit(parallel=True, fastmath=True)
def _hyland_wexler_vectorised(
    temp_k: npt.ArrayLike,
    A: float,
    B: float,
    C: float,
    D: float,
    E: float,
    F: float,
    G: float,
) -> npt.ArrayLike:
    """Calculate saturation vapor pressure arrays using Hyland-Wexler equation.

    Parameters
    ----------
    temp_k : array_like
        Array of temperatures in Kelvin.
    A : float
        Coefficient for 1/T term [K].
    B : float
        Constant coefficient [dimensionless].
    C : float
        Coefficient for T term [K⁻¹].
    D : float
        Coefficient for T² term [K⁻²].
    E : float
        Coefficient for T³ term [K⁻³].
    F : float
        Coefficient for T⁴ term [K⁻⁴].
    G : float
        Coefficient for ln(T) term [dimensionless].

    Returns
    -------
    ndarray
        Saturation vapor pressure in hPa.
    """
    n = len(temp_k)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        result[i] = _hyland_wexler_scalar(temp_k[i], A, B, C, D, E, F, G)
    return result
