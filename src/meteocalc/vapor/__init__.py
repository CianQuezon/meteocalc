from .core import Vapor

from ._enums import EquationName, SurfaceType


from ._vapor_equations import(
    VaporEquation,
    BoltonEquation,
    GoffGratchEquation,
    HylandWexlerEquation
)


__all__ = [
    'Vapor',

    'EquationName',
    'SurfaceType',

    'VaporEquation',
    'BoltonEquation',
    'GoffGratchEquation',
    'HylandWexlerEquation'
]

__version__ = '0.1.0'
__author__ = 'Cian Quezon'