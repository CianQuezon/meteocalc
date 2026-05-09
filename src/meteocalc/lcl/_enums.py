"""
Enumerations for LCL equation identification and calculation method classification.

Author: Cian Quezon
"""

from enum import Enum


class LclEquationName(Enum):
    """
    Identifiers for available LCL equation implementations.

    Attributes
    ----------
    BOLTON : str
        Bolton closed-form lcl approximation.
    ITERATIVE : str
        Uses iterative root-finding solver to find the lcl.
        Accuracy depends on the chosen ``VaporEquation``.
    """

    BOLTON = "bolton"
    ITERATIVE = "iterative"


class CalculationMethod(Enum):
    """
    Classification of LCL calculation approach.

    Attributes
    ----------
    APPROXIMATION : str
        Closed-form analytical approximation — single formula evaluation,
        no iteration.
    SOLVER : str
        Iterative root-finding method — converges to a precise root of
        the LCL objective function. Used by iterative LCL implementations via rapid-roots.

    """

    APPROXIMATION = "approximation"
    SOLVER = "solver"
