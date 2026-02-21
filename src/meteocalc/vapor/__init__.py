from meteocalc.shared._shared_enums import SurfaceType
from ._enums import VaporEquationName
from ._vapor_equations import (
    BoltonEquation,
    GoffGratchEquation,
    HylandWexlerEquation,
    VaporEquation,
)
from .core import Vapor

__all__ = [
    "Vapor",
    "VaporEquationName",
    "SurfaceType",
    "VaporEquation",
    "BoltonEquation",
    "GoffGratchEquation",
    "HylandWexlerEquation",
]

__version__ = "0.1.0"
__author__ = "Cian Quezon"
