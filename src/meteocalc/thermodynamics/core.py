"""
Public interface for thermodynamic potential temperature calculations.
"""

from typing import Optional, Union

import numpy.typing as npt

from meteocalc.shared._enum_tools import parse_enum
from meteocalc.thermodynamics._enums import PotentialTemperatureMode
from meteocalc.thermodynamics._thermodynamic_equation import (
    BoltonPotentialTempEquation,
    PoissonEquation,
)
from meteocalc.thermodynamics._types import POTENTIAL_TEMP_REGISTRY


class Thermodynamics:
    """
    Static-method namespace for thermodynamic potential temperature calculations.

    Three potential temperature variants are provided, all temperatures in
    **Kelvin** and pressure in **hPa**:

    - **Dry** (``mode='dry'``): Poisson's equation θ = T(p₀/p)^κ.
      No moisture inputs required. Throughput ~100 M calculations/second.
    - **LCL** (``mode='lcl'``): Bolton (1980) eq. 24, θL at the lifting
      condensation level. Requires LCL temperature, vapour pressure,
      and mixing ratio.
    - **Equivalent** (``mode='equivalent'``): Bolton (1980) eq. 39, θE.
      Same inputs as LCL. Returns nan for unphysical exponential terms.

    See Also
    --------
    get_potential_temperature : Unified dispatcher for all modes.
    get_dry_potential_temperature : Poisson's equation path.
    get_lcl_potential_temperature : Bolton (1980) eq. 24 path.
    get_equivalent_potential_temperature : Bolton (1980) eq. 39 path.
    """

    @staticmethod
    def get_modes_available() -> list[str]:
        """
        Return names of all available potential temperature modes.

        Returns
        -------
        list[str]
            Mode name strings, one per ``PotentialTemperatureMode`` member.
            Current values: ``['dry', 'lcl', 'equivalent']``.

        Examples
        --------
        >>> Thermodynamics.get_modes_available()
        ['dry', 'lcl', 'equivalent']
        """
        return [mode.value for mode in PotentialTemperatureMode]

    @staticmethod
    def get_equation(mode: Union[str, PotentialTemperatureMode]):
        """
        Return the equation class corresponding to the given mode.

        Parameters
        ----------
        mode : str or PotentialTemperatureMode
            Potential temperature mode. Accepted strings: ``'dry'``,
            ``'lcl'``, ``'equivalent'``.

        Returns
        -------
        type
            Equation class registered under ``mode`` in
            :data:`POTENTIAL_TEMP_REGISTRY`. Instantiate and call
            ``.calculate()`` to run the equation.

        Examples
        --------
        >>> cls = Thermodynamics.get_equation('dry')
        >>> eq = cls()
        >>> eq.calculate(temp_k=288.15, pressure_hpa=850.0)
        301.85...
        """
        mode_enum = parse_enum(mode, PotentialTemperatureMode)
        eq = POTENTIAL_TEMP_REGISTRY[mode_enum]

        if eq is BoltonPotentialTempEquation:
            return eq(mode=mode_enum)
        return eq()

    @staticmethod
    def get_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        mode: Union[str, PotentialTemperatureMode] = "dry",
        lcl_temp_k: Optional[Union[float, npt.ArrayLike]] = None,
        vapor_pressure_hpa: Optional[Union[float, npt.ArrayLike]] = None,
        mixing_ratio: Optional[Union[float, npt.ArrayLike]] = None,
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Compute potential temperature, dispatching to the selected mode.

        Parameters
        ----------
        temp_k : float or array_like of float
            Air temperature in Kelvin.
        pressure_hpa : float or array_like of float
            Atmospheric pressure in hPa.
        mode : str or PotentialTemperatureMode, default ``'dry'``
            Potential temperature variant.
            ``'dry'`` → Poisson's equation (θ).
            ``'lcl'`` → Bolton (1980) eq. 24 (θL).
            ``'equivalent'`` → Bolton (1980) eq. 39 (θE).
        lcl_temp_k : float or array_like of float, optional
            LCL temperature in Kelvin. Required for ``'lcl'`` and
            ``'equivalent'`` modes.
        vapor_pressure_hpa : float or array_like of float, optional
            Water vapour pressure in hPa. Required for ``'lcl'`` and
            ``'equivalent'`` modes.
        mixing_ratio : float or array_like of float, optional
            Water vapour mixing ratio in kg/kg. Required for ``'lcl'``
            and ``'equivalent'`` modes.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Potential temperature in Kelvin.

        Raises
        ------
        ValueError
            If ``mode`` is ``'lcl'`` or ``'equivalent'`` and any of
            ``lcl_temp_k``, ``vapor_pressure_hpa``, or ``mixing_ratio``
            is None.

        Examples
        --------
        >>> Thermodynamics.get_potential_temperature(
        ...     temp_k=288.15, pressure_hpa=850.0
        ... )
        301.85...

        >>> Thermodynamics.get_potential_temperature(
        ...     temp_k=293.15, pressure_hpa=1013.25,
        ...     mode='equivalent',
        ...     lcl_temp_k=283.15,
        ...     vapor_pressure_hpa=12.0,
        ...     mixing_ratio=0.008,
        ... )
        322.59...
        """
        mode_enum = parse_enum(mode, PotentialTemperatureMode)

        if mode_enum == PotentialTemperatureMode.DRY:
            return Thermodynamics.get_dry_potential_temperature(
                temp_k=temp_k, pressure_hpa=pressure_hpa, p0=p0
            )

        missing = [
            name
            for name, val in [
                ("lcl_temp_k", lcl_temp_k),
                ("vapor_pressure_hpa", vapor_pressure_hpa),
                ("mixing_ratio", mixing_ratio),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(f"mode='{mode_enum.value}' requires: {', '.join(missing)}")

        if mode_enum == PotentialTemperatureMode.LCL:
            return Thermodynamics.get_lcl_potential_temperature(
                temp_k=temp_k,
                lcl_temp_k=lcl_temp_k,
                pressure_hpa=pressure_hpa,
                vapor_pressure_hpa=vapor_pressure_hpa,
                mixing_ratio=mixing_ratio,
                p0=p0,
            )

        return Thermodynamics.get_equivalent_potential_temperature(
            temp_k=temp_k,
            lcl_temp_k=lcl_temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=vapor_pressure_hpa,
            mixing_ratio=mixing_ratio,
            p0=p0,
        )

    @staticmethod
    def get_dry_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate dry potential temperature using Poisson's equation.

        Implements θ = T(p₀/p)^κ where κ = Rd/cp = 287.04/1004.0.

        Parameters
        ----------
        temp_k : float or array_like of float
            Air temperature in Kelvin.
        pressure_hpa : float or array_like of float
            Atmospheric pressure in hPa.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Dry potential temperature θ in Kelvin.

        Examples
        --------
        >>> Thermodynamics.get_dry_potential_temperature(
        ...     temp_k=288.15, pressure_hpa=850.0
        ... )
        301.85...
        """
        eq = PoissonEquation()
        return eq.calculate(temp_k=temp_k, pressure_hpa=pressure_hpa, p0=p0)

    @staticmethod
    def get_lcl_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        lcl_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        vapor_pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate potential temperature at the LCL (θL) using Bolton (1980) eq. 24.

        Implements θL = T(p₀/(p−e))^κ · (T/TL)^(0.28w).

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin.
        lcl_temp_k : float or array_like of float
            LCL temperature in Kelvin. Compute via
            :class:`~meteocalc.lcl.core.Lcl`.
        pressure_hpa : float or array_like of float
            Total atmospheric pressure in hPa.
        vapor_pressure_hpa : float or array_like of float
            Water vapour pressure in hPa.
        mixing_ratio : float or array_like of float
            Water vapour mixing ratio in kg/kg.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Potential temperature at LCL (θL) in Kelvin.
        """
        eq = BoltonPotentialTempEquation(mode=PotentialTemperatureMode.LCL)
        return eq.calculate(
            temp_k=temp_k,
            lcl_temp_k=lcl_temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=vapor_pressure_hpa,
            mixing_ratio=mixing_ratio,
            p0=p0,
        )

    @staticmethod
    def get_equivalent_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        lcl_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        vapor_pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate equivalent potential temperature (θE) using Bolton (1980) eq. 39.

        Implements θE = θL · exp[(3376/TL − 2.54) · w · (1 + 0.81w)]
        where θL is Bolton (1980) eq. 24. Returns nan for unphysical
        inputs where the exponential term exceeds 50.

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin.
        lcl_temp_k : float or array_like of float
            LCL temperature in Kelvin. Compute via
            :class:`~meteocalc.lcl.core.Lcl`.
        pressure_hpa : float or array_like of float
            Total atmospheric pressure in hPa.
        vapor_pressure_hpa : float or array_like of float
            Water vapour pressure in hPa.
        mixing_ratio : float or array_like of float
            Water vapour mixing ratio in kg/kg.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Equivalent potential temperature θE in Kelvin.
            nan where exponential term exceeds physical bounds.

        Examples
        --------
        >>> Thermodynamics.get_equivalent_potential_temperature(
        ...     temp_k=293.15, lcl_temp_k=283.15, pressure_hpa=1013.25,
        ...     vapor_pressure_hpa=12.0, mixing_ratio=0.008,
        ... )
        322.59...
        """
        eq = BoltonPotentialTempEquation(mode=PotentialTemperatureMode.EQUIVALENT)
        return eq.calculate(
            temp_k=temp_k,
            lcl_temp_k=lcl_temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=vapor_pressure_hpa,
            mixing_ratio=mixing_ratio,
            p0=p0,
        )
