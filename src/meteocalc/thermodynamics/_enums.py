"""
Enumerations for thermodynamic equation identification and potential
temperature mode selection.

Author: Cian Quezon
"""

from enum import Enum


class ThermodynamicEquationName(Enum):
    """
    Identifiers for thermodynamic equation families.

    Each equation class exposes one of these as its ``name`` attribute,
    grouping related variants (e.g. dry, θL and θE) under one family.

    Attributes
    ----------
    POTENTIAL_TEMP : str
        Potential temperature family. Covers dry (Poisson), LCL and equivalent variants; the
        variant is selected with :class:`PotentialTemperatureMode`.
    """

    POTENTIAL_TEMP = "potential_temp"


class PotentialTemperatureMode(Enum):
    """
    Potential temperature variants available in ``Thermodynamics``.

    Values are the strings accepted by the ``mode`` argument of
    :meth:`Thermodynamics.get_potential_temperature` and
    :meth:`Thermodynamics.get_equation` (case-insensitive).

    Attributes
    ----------
    DRY : str
        Dry potential temperature θ via Poisson's equation.
        Requires temperature and pressure only.
    EQUIVALENT : str
        Equivalent potential temperature θE via Bolton.
        Conserved in both dry and pseudo-adiabatic (moist) processes.
        Requires LCL temperature, vapour pressure and mixing ratio.
    LCL : str
        Potential temperature at the lifting condensation level θL via
        Bolton. Dry potential temperature of the dry-air component, corrected for
        moisture. Requires the same inputs as ``EQUIVALENT``.

    See Also
    --------
    PoissonEquation : Implements ``DRY``.
    BoltonPotentialTempEquation : Implements ``LCL`` and ``EQUIVALENT``.
    """

    DRY = "dry"
    EQUIVALENT = "equivalent"
    LCL = "lcl"
