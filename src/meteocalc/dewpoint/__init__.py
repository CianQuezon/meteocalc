"""
Dew point and frost point calculations for meteocalc.

Public entry point is :class:`Dewpoint`, a static-method namespace that
computes dew point (over water) and frost point (over ice) from air
temperature and relative humidity using one of two strategies:

- Approximation: Magnus-Tetens closed-form formula (Numba JIT)
- Solver: exact numerical inversion of a saturation vapor pressure equation
  (rapid-roots)

All temperatures are in Kelvin and relative humidity is a fraction (0-1).

Examples
--------
>>> from meteocalc.dewpoint import Dewpoint
>>> td = Dewpoint.get_dewpoint(temp_k=293.15, rh=0.6)
>>> tf = Dewpoint.get_frostpoint(temp_k=263.15, rh=0.7, calculation_method="solver")
"""

from meteocalc.shared._shared_enums import SurfaceType

from ._dewpoint_equations import (
    DewPointEquation,
    MagnusDewpointEquation,
    VaporInversionDewpoint,
)
from ._enums import CalculationMethod, DewPointEquationName
from .core import Dewpoint

__all__ = [
    "Dewpoint",
    "DewPointEquationName",
    "CalculationMethod",
    "SurfaceType",
    "DewPointEquation",
    "MagnusDewpointEquation",
    "VaporInversionDewpoint",
]

__version__ = "0.1.0"
__author__ = "Cian Quezon"
