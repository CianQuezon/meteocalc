"""
Equation registries for LCL calculation dispatch.

Three module-level dicts map :class:`~meteocalc.lcl._enums.LclEquationName`
members to their equation classes.  ``Lcl.core`` resolves the user-supplied
name string to an enum via :func:`~meteocalc.shared._enum_tools.parse_enum`,
then does a single ``O(1)`` dict lookup here.

Registries
----------
LCL_SOLVER_REGISTRY : dict
    Solver-strategy equations (Brent root-finding via ``rapid-roots``).
    Keys: :attr:`LclEquationName.ITERATIVE`.
LCL_APPROXIMATION_REGISTRY : dict
    Approximation-strategy equations (closed-form, Numba JIT).
    Keys: :attr:`LclEquationName.BOLTON`.
LCL_EQUATION_MAP : dict
    Union of both registries.  Used by :meth:`Lcl.get_equation` to
    retrieve any equation class by name regardless of strategy.

Adding a new equation
---------------------
1. Implement the JIT scalar/vector functions in ``_jit_equations.py``.
2. Wrap them in a class in ``_lcl_equation.py`` (subclass
   ``LiftingCondensationLevelEquation``).
3. Add a member to :class:`~meteocalc.lcl._enums.LclEquationName`.
4. Register the class in the appropriate dict(s) below.
"""

from meteocalc.lcl._enums import LclEquationName
from meteocalc.lcl._lcl_equation import BoltonLclEquation, IterativeLclEquation

LCL_SOLVER_REGISTRY: dict = {LclEquationName.ITERATIVE: IterativeLclEquation}
"""Solver-strategy registry: ``{LclEquationName → equation class}``."""

LCL_APPROXIMATION_REGISTRY: dict = {LclEquationName.BOLTON: BoltonLclEquation}
"""Approximation-strategy registry: ``{LclEquationName → equation class}``."""

LCL_EQUATION_MAP: dict = {
    LclEquationName.ITERATIVE: IterativeLclEquation,
    LclEquationName.BOLTON: BoltonLclEquation,
}
"""Union registry covering all strategies: ``{LclEquationName → equation class}``."""
