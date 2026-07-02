"""
Equation registry for thermodynamic potential temperature dispatch.

Maps :class:`~meteocalc.thermodynamics._enums.PotentialTemperatureMode`
members to their equation classes. ``Thermodynamics.core`` resolves the
user-supplied mode string to an enum via
:func:`~meteocalc.shared._enum_tools.parse_enum`, then does a single
O(1) dict lookup here.

Registry
--------
POTENTIAL_TEMP_REGISTRY : dict
    ``{PotentialTemperatureMode → equation class}``
    DRY        → :class:`PoissonEquation`
    LCL        → :class:`BoltonPotentialTempEquation`
    EQUIVALENT → :class:`BoltonPotentialTempEquation`

Adding a new potential temperature variant
------------------------------------------
1. Add JIT scalar/vector functions to ``_jit_equations.py``.
2. Add a class (or extend ``BoltonPotentialTempEquation``) in
   ``_thermodynamic_equation.py``.
3. Add a member to :class:`~meteocalc.thermodynamics._enums.PotentialTemperatureMode`.
4. Register it in ``POTENTIAL_TEMP_REGISTRY`` below.
"""

from meteocalc.thermodynamics._enums import PotentialTemperatureMode
from meteocalc.thermodynamics._thermodynamic_equation import (
    BoltonPotentialTempEquation,
    PoissonEquation,
)

POTENTIAL_TEMP_REGISTRY: dict = {
    PotentialTemperatureMode.DRY: PoissonEquation,
    PotentialTemperatureMode.LCL: BoltonPotentialTempEquation,
    PotentialTemperatureMode.EQUIVALENT: BoltonPotentialTempEquation,
}
"""Registry mapping potential temperature mode → equation class."""