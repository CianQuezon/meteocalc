"""
Classes for calculating the lcl. It implements:
- Bolton
- Iterative method

Author: Cian Quezon
"""

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

from meteocalc.lcl._enums import CalculationMethod, LclEquationName
from meteocalc.lcl._jit_equations import _bolton_lcl_scalar, _bolton_lcl_vectorised


class LiftingCondensationLevelEquation(ABC):
    """
    Abstract base class for Lifting Condensation Level (LCL) calculations.

    Provides a common interface for different LCL calculation methods,
    handling input broadcasting, scalar/array dispatch, and shape
    restoration. Subclasses implement specific LCL equations (Bolton
    closed-form, iterative Brent solver) with a consistent public API.

    Attributes
    ----------
    name : LclEquationName
        Identifier for the LCL equation type.
    calculation_method : CalculationMethod
        Whether the equation uses a closed-form or iterative method.

    Notes
    -----
    All public LCL methods accept scalar or array inputs transparently:

    - Scalar inputs (``float``) return scalar outputs (``float``).
    - Array inputs (``ndarray``) return arrays of the same shape.
    - Mixed scalar/array inputs are broadcast to a common shape.

    Subclasses must implement ``calculate`` using
    ``_dispatch_scalar_or_vectorised`` to inherit this behaviour
    automatically.

    Examples
    --------
    Subclass implementation pattern:

    >>> class BoltonLCL(LiftingCondensationLevelEquation):
    ...     name = LclEquationName.BOLTON
    ...     calculation_method = CalculationMethod.CLOSED_FORM
    ...
    ...     def calculate(self, temp_k, dewpoint_temp_k, pressure_hpa):
    ...         return self._dispatch_scalar_or_vectorised(
    ...             temp_k=temp_k,
    ...             dewpoint_temp_k=dewpoint_temp_k,
    ...             pressure_hpa=pressure_hpa,
    ...             scalar_func=_bolton_lcl_scalar,
    ...             vector_func=_bolton_lcl_vectorised,
    ...         )
    """

    name: LclEquationName
    calculation_method: CalculationMethod

    def _broadcast_input(
        self,
        temp_k: Union[float, npt.NDArray],
        dewpoint_temp_k: Union[float, npt.NDArray],
        pressure_hpa: Union[float, npt.NDArray],
    ) -> tuple[
        Union[float, npt.NDArray],
        Union[float, npt.NDArray],
        Union[float, npt.NDArray],
        bool,
    ]:
        """
        Broadcast and validate inputs to a common shape.

        Converts all inputs to float64 arrays and broadcasts them to
        a consistent shape. Scalar inputs are detected before broadcasting
        so the caller can return scalar outputs.

        Parameters
        ----------
        temp_k : float or ndarray
            Surface air temperature in Kelvin.
        dewpoint_temp_k : float or ndarray
            Surface dewpoint temperature in Kelvin.
        pressure_hpa : float or ndarray
            Surface pressure in hPa.

        Returns
        -------
        temp_k : ndarray of float64
        dewpoint_temp_k : ndarray of float64
        pressure_hpa : ndarray of float64
        was_scalar : bool
            True if all inputs were scalar
        """

        temp_k = np.asarray(temp_k, dtype=np.float64)
        dewpoint_temp_k = np.asarray(dewpoint_temp_k, dtype=np.float64)
        pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)

        was_scalar = (
            temp_k.ndim == 0 and dewpoint_temp_k.ndim == 0 and pressure_hpa.ndim == 0
        )

        temp_k, dewpoint_temp_k, pressure_hpa = np.broadcast_arrays(
            temp_k, dewpoint_temp_k, pressure_hpa
        )

        temp_k = np.ascontiguousarray(temp_k)
        dewpoint_temp_k = np.ascontiguousarray(dewpoint_temp_k)
        pressure_hpa = np.ascontiguousarray(pressure_hpa)

        return temp_k, dewpoint_temp_k, pressure_hpa, was_scalar

    def _dispatch_scalar_or_vectorised(
        self,
        temp_k: Union[float, npt.ArrayLike],
        dewpoint_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        scalar_func: Callable[[float, float, float], tuple],
        vector_func: Callable[[npt.NDArray, npt.NDArray, npt.NDArray], tuple],
    ) -> tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Dispatch to scalar or vectorised LCL function based on input shape.

        Parameters
        ----------
        temp_k : float or array_like
            Surface air temperature in Kelvin.
        dewpoint_temp_k : float or array_like
            Surface dewpoint temperature in Kelvin.
        pressure_hpa : float or array_like
            Surface pressure in hPa.
        scalar_func : callable
            JIT scalar function — called when all inputs are scalar.
        vector_func : callable
            JIT vectorised function — called when any input is an array.

        Returns
        -------
        lcl_temp_k : float or ndarray
            LCL temperature in Kelvin.
        lcl_pressure_hpa : float or ndarray
            LCL pressure in hPa.
        """

        temp_k, dewpoint_temp_k, pressure_hpa, was_scalar = self._broadcast_input(
            temp_k=temp_k, dewpoint_temp_k=dewpoint_temp_k, pressure_hpa=pressure_hpa
        )

        if was_scalar:
            temp_k = float(temp_k.item())
            dewpoint_temp_k = float(dewpoint_temp_k.item())
            pressure_hpa = float(pressure_hpa.item())

            lcl_temp_k, lcl_pressure_hpa = scalar_func(
                temp_k=temp_k,
                dewpoint_temp_k=dewpoint_temp_k,
                pressure_hpa=pressure_hpa,
            )

            return lcl_temp_k, lcl_pressure_hpa

        else:
            temp_k_original_shape = temp_k.shape

            temp_k_flatten = temp_k.flatten()
            dewpoint_temp_k_flatten = dewpoint_temp_k.flatten()
            pressure_hpa_flatten = pressure_hpa.flatten()

            lcl_temp_k, lcl_pressure_hpa = vector_func(
                temp_k=temp_k_flatten,
                dewpoint_temp_k=dewpoint_temp_k_flatten,
                pressure_hpa=pressure_hpa_flatten,
            )

            return lcl_temp_k.reshape(temp_k_original_shape), lcl_pressure_hpa.reshape(
                temp_k_original_shape
            )

    @abstractmethod
    def calculate(
        self,
        temp_k: Union[float, npt.ArrayLike],
        dewpoint_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
    ) -> Union[tuple[float, float], tuple[npt.NDArray, npt.NDArray]]:
        """
        Calculate the Lifting Condensation Level (LCL) temperature and pressure.

        Finds the temperature and pressure at which an air parcel becomes
        saturated when lifted dry-adiabatically from the surface. Scalar
        inputs return scalar outputs; array inputs return arrays of the
        same shape.

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin. Must be greater than
            ``dewpoint_temp_k`` for all parcels — a saturated parcel
            (T == Td) has its LCL at the surface.
        dewpoint_temp_k : float or array_like of float
            Surface dewpoint temperature in Kelvin. Must be less than
            ``temp_k`` for all parcels.
        pressure_hpa : float or array_like of float
            Surface pressure in hectopascals (hPa). Typical range
            850–1050 hPa for surface observations.

        Returns
        -------
        lcl_temp_k : float or ndarray of float64
            Temperature at the LCL in Kelvin. Shape matches the
            broadcast shape of the inputs.
        lcl_pressure_hpa : float or ndarray of float64
            Pressure at the LCL in hectopascals (hPa). Shape matches
            the broadcast shape of the inputs.

        Examples
        --------
        Scalar input:

        >>> lcl_temp, lcl_pressure = equation.calculate(
        ...     temp_k=293.15,
        ...     dewpoint_temp_k=285.15,
        ...     pressure_hpa=1013.25,
        ... )
        >>> print(f"T_LCL = {lcl_temp:.2f} K  P_LCL = {lcl_pressure:.2f} hPa")
        T_LCL = 283.35 K  P_LCL = 917.23 hPa

        Array input:

        >>> import numpy as np
        >>> temp_arr     = np.array([293.15, 303.15, 283.15])
        >>> dewpoint_arr = np.array([285.15, 298.15, 278.15])
        >>> pressure_arr = np.full(3, 1013.25)
        >>> lcl_temps, lcl_pressures = equation.calculate(
        ...     temp_k=temp_arr,
        ...     dewpoint_temp_k=dewpoint_arr,
        ...     pressure_hpa=pressure_arr,
        ... )
        >>> print(lcl_temps)
        [283.35 296.94 277.06]
        """
        pass


class BoltonLclEquation(LiftingCondensationLevelEquation):
    """
    LCL temperature and pressure using Bolton (1980) closed-form approximation.

    Implements Bolton (1980) eq. 15 for LCL temperature and the dry
    adiabatic Poisson relation for LCL pressure. This is the fastest
    LCL method in meteocalc — a single formula evaluation with no
    iteration — at the cost of an approximation error of ~0.1–2 K
    depending on dewpoint depression.

    Valid for surface temperatures in [233.15, 323.15] K (-40°C to 50°C)
    and dewpoint depressions up to ~30 K, as stated in Bolton (1980).
    Results outside this range are extrapolations and may be inaccurate.

    Attributes
    ----------
    name : LclEquationName
        ``LclEquationName.BOLTON``
    calculation_method : CalculationMethod
        ``CalculationMethod.APPROXIMATION``

    Examples
    --------
    Scalar input:

    >>> from meteocalc.lcl._lcl_equation import BoltonLclEquation
    >>> eq = BoltonLclEquation()
    >>> lcl_temp, lcl_pressure = eq.calculate(
    ...     temp_k=293.15,
    ...     dewpoint_temp_k=285.15,
    ...     pressure_hpa=1013.25,
    ... )
    >>> print(f"T_LCL = {lcl_temp:.2f} K  ({lcl_temp - 273.15:.2f} °C)")
    T_LCL = 283.35 K  (10.20 °C)
    >>> print(f"P_LCL = {lcl_pressure:.2f} hPa")
    P_LCL = 917.23 hPa

    Array input:

    >>> import numpy as np
    >>> temp_arr     = np.array([293.15, 303.15, 283.15])
    >>> dewpoint_arr = np.array([285.15, 298.15, 278.15])
    >>> pressure_arr = np.full(3, 1013.25)
    >>> lcl_temps, lcl_pressures = eq.calculate(
    ...     temp_k=temp_arr,
    ...     dewpoint_temp_k=dewpoint_arr,
    ...     pressure_hpa=pressure_arr,
    ... )
    >>> print(lcl_temps)
    [283.35 296.94 277.06]
    """

    name: LclEquationName = LclEquationName.BOLTON
    calculation_method: CalculationMethod = CalculationMethod.APPROXIMATION

    def calculate(self, temp_k, dewpoint_temp_k, pressure_hpa):
        lcl_temp_k, lcl_pressure_hpa = self._dispatch_scalar_or_vectorised(
            temp_k=temp_k,
            dewpoint_temp_k=dewpoint_temp_k,
            pressure_hpa=pressure_hpa,
            scalar_func=_bolton_lcl_scalar,
            vector_func=_bolton_lcl_vectorised,
        )
        return lcl_temp_k, lcl_pressure_hpa
