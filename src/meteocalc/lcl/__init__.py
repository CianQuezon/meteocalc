"""
Lifting Condensation Level (LCL) calculations for meteocalc.

Public entry point is :class:`Lcl`, a static-method namespace that returns the
LCL as a ``(temperature, pressure)`` pair using one of two strategies:

- Approximation: Bolton (1980) closed-form formula (Numba JIT)
- Solver: iterative root-finding on the saturation mixing ratio (rapid-roots)

All temperatures are in Kelvin and pressures in hPa.

Examples
--------
>>> from meteocalc.lcl import Lcl
>>> t_lcl, p_lcl = Lcl.get_lcl(
...     temp_k=293.15, dewpoint_temp_k=285.15, pressure_hpa=1013.25
... )
"""

from meteocalc.shared._shared_enums import SurfaceType

from ._enums import CalculationMethod, LclEquationName
from ._lcl_equation import (
    BoltonLclEquation,
    IterativeLclEquation,
    LiftingCondensationLevelEquation,
)
from .core import Lcl

__all__ = [
    "Lcl",
    "LclEquationName",
    "CalculationMethod",
    "SurfaceType",
    "LiftingCondensationLevelEquation",
    "BoltonLclEquation",
    "IterativeLclEquation",
]

__version__ = "0.1.0"
__author__ = "Cian Quezon"
