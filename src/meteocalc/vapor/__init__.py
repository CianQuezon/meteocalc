from ._enums import EquationName, SurfaceType
from ._vapor_equations import (
    BoltonEquation,
    GoffGratchEquation,
    HylandWexlerEquation,
    VaporEquation,
)
from .core import Vapor

__all__ = [
    "Vapor",
    "EquationName",
    "SurfaceType",
    "VaporEquation",
    "BoltonEquation",
    "GoffGratchEquation",
    "HylandWexlerEquation",
]

__version__ = "0.1.0"
__author__ = "Cian Quezon"
