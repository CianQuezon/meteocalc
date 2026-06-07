"""
Public interface for Lifting Condensation Level (LCL) calculations.
"""

import warnings
from typing import Optional, Union

import numpy.typing as npt

from meteocalc.lcl._enums import CalculationMethod, LclEquationName
from meteocalc.lcl._types import (
    LCL_APPROXIMATION_REGISTRY,
    LCL_EQUATION_MAP,
    LCL_SOLVER_REGISTRY,
)
from meteocalc.shared._enum_tools import parse_enum
from meteocalc.shared._shared_enums import SurfaceType
from meteocalc.vapor._enums import VaporEquationName
from meteocalc.vapor.core import Vapor


class Lcl:
    """
    Static-method namespace for Lifting Condensation Level (LCL) computations.

    The LCL is the level at which an unsaturated surface parcel, lifted
    dry-adiabatically, first becomes saturated.  It is expressed as a
    temperature (K) and pressure (hPa) pair and serves as the primary
    estimate of convective cloud-base height in operational meteorology.

    Two calculation strategies are provided:

    - **Approximation** (``calculation_method='approximation'``): Bolton
      (1980) closed-form formula evaluated via Numba JIT.
      Throughput ~100 M calculations/second; accuracy ~0.01 K.
    - **Solver** (``calculation_method='solver'``): Numerical inversion of
      the saturation mixing-ratio equation using Brent's method
      (``rapid-roots``).  Throughput ~5–6 M calculations/second;
      accuracy limited only by the chosen vapour pressure equation
      (typically ±0.001 K).

    All temperatures are in **Kelvin**; pressure is in **hPa**.
    Relative humidity, where needed, is a **fraction (0–1)**.

    See Also
    --------
    get_lcl : Unified dispatcher for both strategies.
    get_lcl_using_approximation : Bolton (1980) closed-form path.
    get_lcl_using_solver : Brent root-finding path.
    """

    @staticmethod
    def get_equations_available() -> list[str]:
        """
        Return names of all available LCL equation implementations.

        Returns
        -------
        list[str]
            Equation name strings, one per ``LclEquationName`` member.
            Current values: ``['bolton', 'iterative']``.

        Examples
        --------
        >>> Lcl.get_equations_available()
        ['bolton', 'iterative']

        See Also
        --------
        get_lcl : Calculate LCL using a named equation.
        """
        equation_list = []

        for equation in LclEquationName:
            equation_name = equation.value
            equation_list.append(equation_name)
        return equation_list

    @staticmethod
    def get_equation(lcl_equation_name: str):
        """
        Return the equation corresponding to the given name.

        Parameters
        ----------
        lcl_equation_name : str
            The name of the equation to retrieve; must be a valid member of
            :class:`LclEquationName`.

        Returns
        -------
        type
            Equation class registered under ``lcl_equation_name`` in
            :data:`LCL_EQUATION_MAP`.  Instantiate and call ``.calculate()``
            to run the equation.

        Examples
        --------
        >>> cls = Lcl.get_equation("bolton")
        >>> eq = cls()
        >>> t_lcl, p_lcl = eq.calculate(temp_k=293.15, dewpoint_temp_k=285.15, pressure_hpa=1013.25)
        """
        lcl_equation_enum = parse_enum(lcl_equation_name, LclEquationName)
        equation_selected = LCL_EQUATION_MAP[lcl_equation_enum]

        return equation_selected

    @staticmethod
    def get_lcl(
        temp_k: Union[float, npt.ArrayLike],
        dewpoint_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Optional[Union[float, npt.ArrayLike]] = None,
        lcl_equation_name: str = "bolton",
        vapor_equation_name: str = "goff_gratch",
        calculation_method: Union[str, CalculationMethod] = "approximation",
        surface_type: Union[str, SurfaceType] = "automatic",
    ) -> tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Compute the LCL temperature and pressure, dispatching to approximation or solver.

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin.
        dewpoint_temp_k : float or array_like of float
            Surface dewpoint temperature in Kelvin.
        pressure_hpa : float or array_like of float
            Surface pressure in hectopascals (hPa).
        mixing_ratio : float or array_like of float, optional
            Surface water vapour mixing ratio in kg/kg.  Required when
            ``calculation_method='solver'``; unused for ``'approximation'``.
        lcl_equation_name : str, default ``'bolton'``
            LCL equation to use.  ``'bolton'`` selects the Bolton (1980)
            closed-form approximation; ``'iterative'`` selects the Brent
            root-finding solver.  Passing ``'bolton'`` with
            ``calculation_method='solver'`` raises a ``UserWarning`` and
            falls back to ``'iterative'``.
        vapor_equation_name : str, default ``'goff_gratch'``
            Saturation vapour pressure equation used in the solver objective
            function.  Options: ``'goff_gratch'``, ``'hyland_wexler'``,
            ``'bolton'``.  Ignored when ``calculation_method='approximation'``.
        calculation_method : str or CalculationMethod, default ``'approximation'``
            Strategy to use.  ``'approximation'`` calls
            :meth:`get_lcl_using_approximation`; ``'solver'`` calls
            :meth:`get_lcl_using_solver`.
        surface_type : str or SurfaceType, default ``'automatic'``
            Phase surface for vapour pressure constants.
            ``'automatic'`` selects ice below 273.15 K, water otherwise.
            Ignored when ``calculation_method='approximation'``.

        Returns
        -------
        lcl_temp_k : float or ndarray of float64
            LCL temperature in Kelvin.
        lcl_pressure_hpa : float or ndarray of float64
            LCL pressure in hPa.

        Examples
        --------
        >>> # Approximation (default)
        >>> t_lcl, p_lcl = Lcl.get_lcl(temp_k=293.15, dewpoint_temp_k=285.15, pressure_hpa=1013.25)
        >>> print(f"T_LCL = {t_lcl:.2f} K  P_LCL = {p_lcl:.2f} hPa")
        T_LCL = 283.35 K  P_LCL = 899.56 hPa

        See Also
        --------
        get_lcl_using_approximation : Bolton (1980) closed-form path.
        get_lcl_using_solver : Brent root-finding path.
        """
        calculation_method_enum = parse_enum(calculation_method, CalculationMethod)

        if calculation_method_enum == CalculationMethod.APPROXIMATION:
            return Lcl.get_lcl_using_approximation(
                temp_k=temp_k,
                dewpoint_temp_k=dewpoint_temp_k,
                pressure_hpa=pressure_hpa,
                lcl_equation_name=lcl_equation_name,
            )
        elif calculation_method_enum == CalculationMethod.SOLVER:
            if lcl_equation_name == "bolton":
                warnings.warn(
                    "The 'bolton' equation is an approximation and cannot be used with calculation_method='solver'. Overriding to 'iterative' solver.",
                    UserWarning,
                    stacklevel=2,
                )
                lcl_equation_name = "iterative"

            return Lcl.get_lcl_using_solver(
                temp_k=temp_k,
                dewpoint_temp_k=dewpoint_temp_k,
                pressure_hpa=pressure_hpa,
                mixing_ratio=mixing_ratio,
                solver_name=lcl_equation_name,
                vapor_equation_name=vapor_equation_name,
                surface_type=surface_type,
            )

    @staticmethod
    def get_lcl_using_approximation(
        temp_k: Union[float, npt.ArrayLike],
        dewpoint_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        lcl_equation_name: Union[str, LclEquationName] = "bolton",
    ) -> tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Calculate LCL temperature and pressure using an approximation.

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin.
        dewpoint_temp_k : float or array_like of float
            Surface dewpoint temperature in Kelvin.
        pressure_hpa : float or array_like of float
            Surface pressure in hectopascals (hPa).
        lcl_equation_name : str or LclEquationName, default ``'bolton'``
            Name of equation to use for LCL approximation calculation.
        Returns
        -------
        lcl_temp_k : float or ndarray of float64
            LCL temperature in Kelvin.
        lcl_pressure_hpa : float or ndarray of float64
            LCL pressure in hPa.
        """
        approximation_equation_enum = parse_enum(lcl_equation_name, LclEquationName)
        approximation_equation = LCL_APPROXIMATION_REGISTRY[approximation_equation_enum]
        lcl_equation = approximation_equation()

        results = lcl_equation.calculate(
            temp_k=temp_k, dewpoint_temp_k=dewpoint_temp_k, pressure_hpa=pressure_hpa
        )
        return results

    @staticmethod
    def get_lcl_using_solver(
        temp_k: Union[float, npt.ArrayLike],
        dewpoint_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Union[float, npt.ArrayLike],
        solver_name: Union[str, LclEquationName] = "iterative",
        vapor_equation_name: Union[str, VaporEquationName] = "goff_gratch",
        surface_type: Union[str, SurfaceType] = "automatic",
    ) -> tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Calculate LCL temperature and pressure using an iterative Brent solver.

        Numerically inverts the saturation mixing ratio equation to find the
        temperature at which a surface parcel becomes saturated when lifted
        dry-adiabatically. LCL pressure is derived from the result via the
        Poisson relation. Accuracy is limited only by the chosen vapor pressure
        equation (typically ±0.001 K).

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin. Must exceed ``dewpoint_temp_k``
            for all parcels.
        dewpoint_temp_k : float or array_like of float
            Surface dewpoint temperature in Kelvin.
        pressure_hpa : float or array_like of float
            Surface pressure in hectopascals (hPa).
        mixing_ratio : float or array_like of float
            Surface water vapour mixing ratio in kg/kg. Compute at
            ``dewpoint_temp_k`` using the same ``vapor_equation_name`` for
            physical consistency:

                e = vapor_equation.calculate(dewpoint_temp_k)
                w = eps * e / (pressure_hpa - e)

        solver_name : str or LclEquationName, default ``'iterative'``
            Root-finding implementation to use. Currently only
            ``'iterative'`` (Brent + bisection backup) is supported.
        vapor_equation_name : str or VaporEquationName, default ``'goff_gratch'``
            Saturation vapour pressure formulation used in the objective
            function. Options: ``'goff_gratch'``, ``'hyland_wexler'``,
            ``'bolton'``.
        surface_type : str or SurfaceType, default ``'automatic'``
            Phase surface for vapour pressure constants.
            ``'automatic'`` selects ice below 273.15 K, water otherwise.

        Returns
        -------
        lcl_temp_k : float or ndarray of float64
            LCL temperature in Kelvin.
        lcl_pressure_hpa : float or ndarray of float64
            LCL pressure in hPa.

        Examples
        --------
        >>> from meteocalc.shared.constants import eps
        >>> from meteocalc.vapor.core import Vapor
        >>> vapor_eq = Vapor.get_equation('goff_gratch', phase='water')
        >>> e = float(vapor_eq.calculate(285.15))
        >>> w = eps * e / (1013.25 - e)
        >>> lcl_temp, lcl_pres = Lcl.get_lcl_using_solver(
        ...     temp_k=293.15,
        ...     dewpoint_temp_k=285.15,
        ...     pressure_hpa=1013.25,
        ...     mixing_ratio=w,
        ... )
        >>> print(f"T_LCL = {lcl_temp:.4f} K  P_LCL = {lcl_pres:.4f} hPa")
        T_LCL = 283.3605 K  P_LCL = 899.7022 hPa

        See Also
        --------
        get_lcl : Dispatcher that routes to approximation or solver.
        meteocalc.lcl._solver_method.get_lcl_using_solver : Low-level solver.
        """
        lcl_solver_equation_enum = parse_enum(solver_name, LclEquationName)
        lcl_solver_equation = LCL_SOLVER_REGISTRY[lcl_solver_equation_enum]

        vapor_equation = Vapor.get_equation(
            equation=vapor_equation_name, phase=surface_type
        )

        solver = lcl_solver_equation()
        results = solver.calculate(
            temp_k=temp_k,
            dewpoint_temp_k=dewpoint_temp_k,
            pressure_hpa=pressure_hpa,
            mixing_ratio=mixing_ratio,
            vapor_equation=vapor_equation,
        )
        return results
