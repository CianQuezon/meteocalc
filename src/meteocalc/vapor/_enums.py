"""
Enums to use for vapor saturation equations.

Implements:
 -Phase Mode
 -Surface Type

Author: Cian Quezon
"""
from enum import Enum


class SurfaceType(Enum):
    """
    Surface type for vapor pressure calculations.

    Different equations use different constants depending on surface types.

    Args:
        - AUTOMATIC = Automatically chooses ice and water surface type depending on temperature
        - ICE = Uses ice constants for vapor saturation equation
        - WATER = Uses water constants for vapor saturation equation
    """

    AUTOMATIC = "automatic"
    ICE = "ice"
    WATER = "water"


class EquationName(Enum):
    """
    Equation names for the saturation vapor pressure equations.

    Args:
        - BOLTON
        - GOFF_GRATCH
        - HYLAND_WEXLER
    """

    BOLTON = "bolton"
    GOFF_GRATCH = "goff_gratch"
    HYLAND_WEXLER = "hyland_wexler"
