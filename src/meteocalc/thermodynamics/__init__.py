"""
Thermodynamic calculations for meteocalc.

Public entry point is :class:`Thermodynamics`, a static-method namespace for
potential temperature:

- Dry potential temperature θ (Poisson's equation)
- Potential temperature at the LCL θL (Bolton 1980, eq. 24)
- Equivalent potential temperature θE (Bolton 1980, eq. 39)

All temperatures are in Kelvin, pressures in hPa and mixing ratios in kg/kg.

Examples
--------
>>> from meteocalc.thermodynamics import Thermodynamics
>>> Thermodynamics.get_dry_potential_temperature(temp_k=288.15, pressure_hpa=850.0)
301.85...
"""

from ._enums import PotentialTemperatureMode, ThermodynamicEquationName
from ._thermodynamic_equation import (
    BoltonPotentialTempEquation,
    PoissonEquation,
    PotentialTemperatureEquation,
)
from .core import Thermodynamics

__all__ = [
    "Thermodynamics",
    "PotentialTemperatureMode",
    "ThermodynamicEquationName",
    "PotentialTemperatureEquation",
    "PoissonEquation",
    "BoltonPotentialTempEquation",
]

__version__ = "0.1.0"
__author__ = "Cian Quezon"
