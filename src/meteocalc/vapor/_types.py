"""
Store the vapor registries and other config types.

Author: Cian Quezon
"""

from meteocalc.vapor._enums import VaporEquationName
from meteocalc.vapor._vapor_equations import BoltonEquation, GoffGratchEquation, HylandWexlerEquation

EQUATION_REGISTRY = {
    VaporEquationName.BOLTON: BoltonEquation,
    VaporEquationName.GOFF_GRATCH: GoffGratchEquation,
    VaporEquationName.HYLAND_WEXLER: HylandWexlerEquation,
}

