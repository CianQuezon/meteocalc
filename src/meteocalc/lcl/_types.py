"""
Docstring for meteocalc.lcl._types
"""

from meteocalc.lcl._lcl_equation import BoltonLclEquation, IterativeLclEquation
from meteocalc.lcl._enums import LclEquationName

LCL_SOLVER_REGISTRY = {LclEquationName.ITERATIVE: IterativeLclEquation}
LCL_APPROXIMATION_REGISTRY = {LclEquationName.BOLTON: BoltonLclEquation}
LCL_EQUATION_MAP = {LclEquationName.ITERATIVE: IterativeLclEquation, LclEquationName.BOLTON: BoltonLclEquation}