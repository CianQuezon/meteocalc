"""
Constants for different Saturation Vapor Equations.

Implements:
- Goff & Gratch

Author: Cian Quezon

References:
    https://cires1.colorado.edu/~voemel/vp.html
"""

from typing import NamedTuple

import numpy as np


class GoffGratchConstants(NamedTuple):
    """Constants for Goff-Gratch saturation vapor pressure formula.

    Attributes
    ----------
    T_ref : float
        Reference temperature in Kelvin.
    A : float
        Coefficient for (T_ref/T - 1) term.
    B : float
        Coefficient for log₁₀(T_ref/T) term.
    C : float
        Coefficient for exponential term with C_exp.
    C_exp : float
        Exponent coefficient for C term.
    D : float
        Coefficient for exponential term with D_exp.
    D_exp : float
        Exponent coefficient for D term.
    log_p_ref : float
        Log₁₀ of reference pressure.
    """

    T_ref: float
    A: float
    B: float
    C: float
    C_exp: float
    D: float
    D_exp: float
    log_p_ref: float


GOFF_GRATCH_WATER = GoffGratchConstants(
    T_ref=373.16,
    A=-7.90298,
    B=5.02808,
    C=-1.3816e-7,
    C_exp=11.344,
    D=8.1328e-3,
    D_exp=-3.49149,
    log_p_ref=np.log10(1013.246),
)

GOFF_GRATCH_ICE = GoffGratchConstants(
    T_ref=273.16,
    A=-9.09718,
    B=-3.56654,
    C=0.876793,
    C_exp=0,
    D=0,
    D_exp=0,
    log_p_ref=np.log10(6.1071),
)


class HylandWexlerConstants(NamedTuple):
    """Constants for Hyland-Wexler saturation vapor pressure formula.

    Attributes
    ----------
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
    """


    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float


HYLAND_WEXLER_WATER = HylandWexlerConstants(
    A=-0.58002206e4,
    B=0.13914993e1,
    C=-0.48640239e-1,
    D=0.41764768e-4,
    E=-0.14452093e-7,
    F=0,
    G=0.65459673e1,
)

HYLAND_WEXLER_ICE = HylandWexlerConstants(
    A=-0.56745359e4,
    B=0.63925247e1,
    C=-0.96778430e-2,
    D=0.62215701e-6,
    E=0.20747825e-8,
    F=-0.94840240e-12,
    G=0.41635019e1,
)
