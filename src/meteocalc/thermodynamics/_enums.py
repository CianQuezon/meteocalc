"""
Docstring for meteocalc.thermodynamics._enums
"""
from enum import Enum


class ThermodynamicEquationName(Enum):
    POTENTIAL_TEMP = "potential_temp"

class PotentialTemperatureMode(Enum):
    STANDARD = "standard"
    EQUIVALENT = "equivalent"
