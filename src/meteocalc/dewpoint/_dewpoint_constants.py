"""
Named tuple dewpoint constants.

Author: Cian Quezon
"""


from typing import NamedTuple

class MagnusDewpointConstants(NamedTuple):
    """
    Magnus-Tetens equation coefficients for dew/frost point calculation.
    
    Named tuple containing the empirical coefficients A and B used in the
    Magnus-Tetens approximation formula. Different coefficients are used
    for water and ice surfaces to account for different vapor pressure
    characteristics.
    
    Attributes
    ----------
    A : float
        Dimensionless coefficient in Magnus formula

    B : float
        Temperature coefficient in Celsius
    """
    A: float
    B: float

MAGNUS_ICE = MagnusDewpointConstants(A=21.875, B=265.5)
MAGNUS_WATER = MagnusDewpointConstants(A=17.27, B=237.7)