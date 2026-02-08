"""
Docstring for meteocalc.dewpoint._enums
"""

from enum import Enum

class DewPointEquationName(Enum):
    """
    Docstring for EquationName
    """
    MAGNUS = "magnus"
    BUCK = "buck"

class CalculationMethod(Enum):
    """
    Docstring for CalculationMethod
    """
    APPROXIMATION = "approximation"
    SOLVER = "solver"