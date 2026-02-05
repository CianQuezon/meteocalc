"""
Docstring for meteocalc.dewpoint._dewpoint_constants
"""


from typing import NamedTuple

class BuckDewpointConstants(NamedTuple):
    """
    Docstring for BuckConstants
    """

    A: float
    B: float
    C: float


BUCK_ICE = BuckDewpointConstants(A=17.966, B=247.15, C=278.5)
BUCK_WATER = BuckDewpointConstants(A=17.368, B=238.88, C=234.5)


class MagnusDewpointConstants(NamedTuple):
    """
    Docstring for MagnusConstants
    """
    A: float
    B: float

MAGNUS_ICE = MagnusDewpointConstants(A=21.875, B=265.5)
MAGNUS_WATER = MagnusDewpointConstants(A=17.27, B=237.7)