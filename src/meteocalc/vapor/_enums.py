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
    
    Attributes
    ----------
    AUTOMATIC : str
        Auto-selects ice/water based on temperature.
    ICE : str
        Uses ice surface constants.
    WATER : str
        Uses water surface constants.
    """

    AUTOMATIC = "automatic"
    ICE = "ice"
    WATER = "water"


class EquationName(Enum):
    """
    Available saturation vapor pressure equation types.
    
    Attributes
    ----------
    BOLTON : str
        Bolton equation.
    GOFF_GRATCH : str
        Goff-Gratch equation.
    HYLAND_WEXLER : str
        Hyland-Wexler equation.
    """

    BOLTON = "bolton"
    GOFF_GRATCH = "goff_gratch"
    HYLAND_WEXLER = "hyland_wexler"
