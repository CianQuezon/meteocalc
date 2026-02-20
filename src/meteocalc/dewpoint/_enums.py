"""
Enums to use for dewpoin equations.

Implements:
 -Calculation Method
 -Dewpoint Equation Name

Author: Cian Quezon
"""

from enum import Enum


class DewPointEquationName(Enum):
    """
    Enumeration of available dew point equation names.

    Defines all supported methods for calculating dew point and frost point,
    including both fast approximation formulas and exact numerical inversion
    methods.

    Attributes
    ----------
    MAGNUS : str
        Magnus-Tetens approximation formula

    VAPOR_INVERSION : str
        Exact numerical inversion of vapor pressure equations

    See Also
    --------
    Dewpoint.get_dewpoint_approximation : Use approximation methods
    Dewpoint.get_dewpoint_solver : Use exact solver
    Dewpoint.get_equations_available : List all available equations
    CalculationMethod : High-level method selection enum
    """

    MAGNUS = "magnus"
    VAPOR_INVERSION = "vapor_inversion"


class CalculationMethod(Enum):
    """
    Enumeration of available calculation methods for dew point.

    Defines the two primary approaches for calculating dew point and
    frost point temperatures: fast approximations and exact numerical solvers.

    Attributes
    ----------
    APPROXIMATION : str
        Fast analytical approximation methods (e.g., Magnus, Buck)

    SOLVER : str
        Exact numerical solver using vapor pressure inversion

    See Also
    --------
    Dewpoint.get_dewpoint : Unified interface using this enum
    Dewpoint.get_frostpoint : Frost point with method selection
    DewPointEquationName : Enumeration of specific equations
    """

    APPROXIMATION = "approximation"
    SOLVER = "solver"
