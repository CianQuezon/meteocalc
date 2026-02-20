"""
Docstring for meteocalc.dewpoint._types
"""

from meteocalc.dewpoint._dewpoint_equations import (
    MagnusDewpointEquation,
    VaporInversionDewpoint,
)
from meteocalc.dewpoint._enums import DewPointEquationName

APPROXIMATION_DEWPOINT_REGISTRY = {DewPointEquationName.MAGNUS: MagnusDewpointEquation}

SOLVER_DEWPOINT_REGISTRY = {
    DewPointEquationName.VAPOR_INVERSION: VaporInversionDewpoint
}
