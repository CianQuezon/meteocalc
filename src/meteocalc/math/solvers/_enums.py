"""
Enums for the solvers to get the root

Author: Cian Quezon
"""
from enum import Enum


class SolverName(Enum):
    """
    Enumeration for the names for different types of solvers implemented.

    Attributes:
     - NEWTON = Newton Raphson root finding method
     - BRENT = Brent root finding method
     - BISECTION = Bisection root finding method
    """
    NEWTON = "newton"
    BRENT = "brent"
    BISECTION = "bisection"

class MethodType(Enum):
    """
    Enumeration for the type of method the solver uses.

    Attributes:
        - OPEN = Solver is an open method which requires a guess and an optional derivative.
        - BRACKETING = Solver is a bracketed method which requires an upper and lower bracket.
        - HYBRID = Solver uses a hybrid approach which combines 2 methods
        - CUSTOM = Customized method defined or created by the user
    """
    OPEN = "open"
    BRACKET = "bracket"
    HYBRID = "hybrid"
    CUSTOM = "custom"
