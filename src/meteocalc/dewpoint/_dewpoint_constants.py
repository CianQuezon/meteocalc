"""
Docstring for meteocalc.dewpoint._dewpoint_constants
"""


from typing import NamedTuple

class MagnusDewpointConstants(NamedTuple):
    """
    Docstring for MagnusConstants
    """
    A: float
    B: float

MAGNUS_ICE = MagnusDewpointConstants(A=21.875, B=265.5)
MAGNUS_WATER = MagnusDewpointConstants(A=17.27, B=237.7)