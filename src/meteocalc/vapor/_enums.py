"""
Enums to use for vapor saturation equations.

Implements:
 -Phase Mode
 -Surface Type

Author: Cian Quezon
"""

from enum import Enum


class VaporEquationName(Enum):
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
