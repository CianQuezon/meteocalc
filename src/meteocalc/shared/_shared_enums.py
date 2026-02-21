"""
Shared enums for different modules

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
