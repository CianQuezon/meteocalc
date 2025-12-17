"""
Constants for different Saturation Vapor Equations.

Implements:
- Goff & Gratch

Author: Cian Quezon

References:
    https://cires1.colorado.edu/~voemel/vp.html
"""
import numpy as np

from typing import NamedTuple

class GoffGratchConstants(NamedTuple):
    """
    A class which stores the constants for Goff Gratch saturation formula.

    Equation:
    log₁₀(eₛ) = -A(Tᵣₑf/T - 1) 
                + B·log₁₀(Tᵣₑf/T)
                - C(10^(C_exp·(1 - T/Tᵣₑf)) - 1)
                + D(10^(D_exp·(Tᵣₑf/T - 1)) - 1)
                + log₁₀(pᵣₑf)

    """
    T_ref: np.float64
    P_ref: np.float64
    A: np.float64
    B: np.float64
    C: np.float64
    C_exp: np.float64
    D: np.float64
    D_exp: np.float64
    log_p_ref: np.float64

GOFF_GRATCH_WATER = GoffGratchConstants(
    T_ref=373.16,
    P_ref=1013.246,
    A=-7.90298,
    B=5.02808,
    C=-1.3816e-7,
    C_exp=11.344,
    D=8.1328e-3,
    D_exp=-3.49149,
    log_p_ref=np.log10(1013.246) 
)

GOFF_GRATCH_ICE = GoffGratchConstants(
    T_ref=273.16,
    P_ref=6.1071,
    A=-9.09718,
    B=-3.56654,
    C=0.876793,
    C_exp=0,
    D=0,
    D_exp=0,
    log_p_ref=np.log10(6.1071)
)


class HylandWexlerConstants(NamedTuple):
    """
    Class which stores the constants for Hyland Wexler saturation formula
    
    Equation:
        ln(eᵢ) = -A/T + B - C·T + D·T² + E·T³ - F·T⁴ + G·ln(T)
    """
    A: np.float64
    B: np.float64
    C: np.float64
    D: np.float64
    E: np.float64
    F: np.float64
    G: np.float64

HYLAND_WEXLER_WATER = HylandWexlerConstants(
    A=-0.58002206e4,
    B=0.13914993e1,
    C=-0.48640239e-1,
    D=0.41764768e-4,
    E=-0.14452093e-7,
    F=0,
    G=0.65459673e1
)

HYLAND_WEXLER_ICE = HylandWexlerConstants(
    A=-0.56745359e4,
    B=0.63925247e1,
    C=-0.96778430e-2,
    D=0.62215701e-6,
    E=0.20747825e-8,
    F=-0.94840240e-12,
    G=0.41635019e1     
)