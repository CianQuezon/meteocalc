"""
Equation classes for potential temperature calculations.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import numpy.typing as npt

from meteocalc.thermodynamics._enums import (
    PotentialTemperatureMode,
    ThermodynamicEquationName,
)
from meteocalc.thermodynamics._jit_equations import (
    _bolton_theta_e_scalar,
    _bolton_theta_e_vectorised,
    _bolton_theta_l_scalar,
    _bolton_theta_l_vectorised,
    _poisson_scalar,
    _poisson_vectorised,
)


class PotentialTemperatureEquation(ABC):
    """
    Abstract base class for potential temperature calculations.

    Provides scalar/array dispatch and input broadcasting for all
    potential temperature variants (dry, θL, θE). Subclasses implement
    specific formulas via ``calculate``.

    Attributes
    ----------
    name : ThermodynamicEquationName
        Equation family identifier.
    mode : PotentialTemperatureMode
        Variant: DRY, LCL (θL), or EQUIVALENT (θE).
    """

    name: ThermodynamicEquationName
    mode: PotentialTemperatureMode

    def _broadcast_input(
        self,
        temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
    ) -> tuple[npt.NDArray, npt.NDArray, bool]:
        """
        Broadcast temp_k and pressure_hpa to a common float64 shape.

        Returns
        -------
        temp_k : ndarray of float64
        pressure_hpa : ndarray of float64
        was_scalar : bool
            True if all inputs were scalar.
        """
        temp_k = np.asarray(temp_k, dtype=np.float64)
        pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)

        was_scalar = temp_k.ndim == 0 and pressure_hpa.ndim == 0

        temp_k, pressure_hpa = np.broadcast_arrays(temp_k, pressure_hpa)
        temp_k = np.ascontiguousarray(temp_k)
        pressure_hpa = np.ascontiguousarray(pressure_hpa)

        return temp_k, pressure_hpa, was_scalar

    def _dispatch_scalar_or_vectorised(
        self,
        was_scalar: bool,
        original_shape: tuple,
        scalar_func: Callable,
        vector_func: Callable,
        **kwargs,
    ) -> Union[float, npt.NDArray]:
        """
        Route to scalar or vectorised JIT function based on input shape.

        Parameters
        ----------
        was_scalar : bool
            True if all inputs were originally scalar.
        original_shape : tuple
            Shape of the broadcast inputs — used to reshape vector output.
        scalar_func : callable
            JIT scalar function.
        vector_func : callable
            JIT vectorised function.
        **kwargs
            Arguments forwarded to scalar_func or vector_func.

        Returns
        -------
        float or ndarray of float64
            Potential temperature. Shape matches broadcast shape of inputs.
        """
        if was_scalar:
            scalar_kwargs = {
                k: float(v.item()) if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()
            }
            return float(scalar_func(**scalar_kwargs))
        else:
            flat_kwargs = {
                k: v.flatten() if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()
            }
            return vector_func(**flat_kwargs).reshape(original_shape)

    @abstractmethod
    def calculate(self, **kwargs) -> Union[float, npt.NDArray]:
        """
        Calculate potential temperature.

        Returns
        -------
        float or ndarray of float64
            Potential temperature in Kelvin.
        """
        pass


class PoissonEquation(PotentialTemperatureEquation):
    """
    Dry potential temperature using Poisson's equation.

    Implements θ = T(p₀/p)^κ where κ = Rd/cp = 287.04/1004.0.

    Attributes
    ----------
    name : ThermodynamicEquationName
        ``ThermodynamicEquationName.POTENTIAL_TEMP``
    mode : PotentialTemperatureMode
        ``PotentialTemperatureMode.DRY``

    Examples
    --------
    >>> eq = PoissonEquation()
    >>> eq.calculate(temp_k=288.15, pressure_hpa=850.0)
    312.09...
    """

    name: ThermodynamicEquationName = ThermodynamicEquationName.POTENTIAL_TEMP
    mode: PotentialTemperatureMode = PotentialTemperatureMode.DRY

    def calculate(
        self,
        temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate dry potential temperature.

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
        """
        temp_k, pressure_hpa, was_scalar = self._broadcast_input(temp_k, pressure_hpa)
        original_shape = temp_k.shape

        return self._dispatch_scalar_or_vectorised(
            was_scalar=was_scalar,
            original_shape=original_shape,
            scalar_func=_poisson_scalar,
            vector_func=_poisson_vectorised,
            temp_k=temp_k,
            p=pressure_hpa,
            p0=p0,
        )


class BoltonPotentialTempEquation(PotentialTemperatureEquation):
    """
    Moist potential temperature using Bolton (1980) equations 24 and 39.

    A single class covering both θL (eq. 24) and θE (eq. 39) — the mode
    argument selects which JIT function is dispatched. Both variants share
    identical inputs, so one broadcast/dispatch implementation serves both.

    Attributes
    ----------
    name : ThermodynamicEquationName
        ``ThermodynamicEquationName.POTENTIAL_TEMP``
    mode : PotentialTemperatureMode
        ``PotentialTemperatureMode.LCL`` → θL (Bolton eq. 24)
        ``PotentialTemperatureMode.EQUIVALENT`` → θE (Bolton eq. 39)

    Examples
    --------
    >>> eq = BoltonPotentialTempEquation(mode=PotentialTemperatureMode.LCL)
    >>> eq.calculate(temp_k=293.15, lcl_temp_k=283.15, pressure_hpa=1013.25,
    ...              vapor_pressure_hpa=12.0, mixing_ratio=0.008)
    """

    name: ThermodynamicEquationName = ThermodynamicEquationName.POTENTIAL_TEMP

    _JIT_MAP = {
        PotentialTemperatureMode.LCL: (
            _bolton_theta_l_scalar,
            _bolton_theta_l_vectorised,
        ),
        PotentialTemperatureMode.EQUIVALENT: (
            _bolton_theta_e_scalar,
            _bolton_theta_e_vectorised,
        ),
    }

    def __init__(
        self, mode: PotentialTemperatureMode = PotentialTemperatureMode.EQUIVALENT
    ) -> None:
        if mode not in self._JIT_MAP:
            raise ValueError(
                f"BoltonPotentialTempEquation does not support mode={mode!r}. Use PoissonEquation for DRY."
            )
        self.mode = mode

    def _broadcast_input(
        self,
        temp_k: Union[float, npt.ArrayLike],
        lcl_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        vapor_pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Union[float, npt.ArrayLike],
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, bool]:
        """
        Broadcast all five moist inputs to a common float64 shape.
        """
        temp_k = np.asarray(temp_k, dtype=np.float64)
        lcl_temp_k = np.asarray(lcl_temp_k, dtype=np.float64)
        pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)
        vapor_pressure_hpa = np.asarray(vapor_pressure_hpa, dtype=np.float64)
        mixing_ratio = np.asarray(mixing_ratio, dtype=np.float64)

        was_scalar = (
            temp_k.ndim == 0
            and lcl_temp_k.ndim == 0
            and pressure_hpa.ndim == 0
            and vapor_pressure_hpa.ndim == 0
            and mixing_ratio.ndim == 0
        )

        temp_k, lcl_temp_k, pressure_hpa, vapor_pressure_hpa, mixing_ratio = (
            np.broadcast_arrays(
                temp_k, lcl_temp_k, pressure_hpa, vapor_pressure_hpa, mixing_ratio
            )
        )

        temp_k = np.ascontiguousarray(temp_k)
        lcl_temp_k = np.ascontiguousarray(lcl_temp_k)
        pressure_hpa = np.ascontiguousarray(pressure_hpa)
        vapor_pressure_hpa = np.ascontiguousarray(vapor_pressure_hpa)
        mixing_ratio = np.ascontiguousarray(mixing_ratio)

        return (
            temp_k,
            lcl_temp_k,
            pressure_hpa,
            vapor_pressure_hpa,
            mixing_ratio,
            was_scalar,
        )

    def calculate(
        self,
        temp_k: Union[float, npt.ArrayLike],
        lcl_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        vapor_pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate moist potential temperature (θL or θE) based on mode.

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin.
        lcl_temp_k : float or array_like of float
            LCL temperature in Kelvin.
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
            θL (Bolton eq. 24) or θE (Bolton eq. 39) in Kelvin.
            θE returns nan where exponential term exceeds physical bounds.
        """
        scalar_func, vector_func = self._JIT_MAP[self.mode]

        (
            temp_k,
            lcl_temp_k,
            pressure_hpa,
            vapor_pressure_hpa,
            mixing_ratio,
            was_scalar,
        ) = self._broadcast_input(
            temp_k, lcl_temp_k, pressure_hpa, vapor_pressure_hpa, mixing_ratio
        )
        original_shape = temp_k.shape

        return self._dispatch_scalar_or_vectorised(
            was_scalar=was_scalar,
            original_shape=original_shape,
            scalar_func=scalar_func,
            vector_func=vector_func,
            temp_k=temp_k,
            lcl_temp_k=lcl_temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=vapor_pressure_hpa,
            mixing_ratio=mixing_ratio,
            p0=p0,
        )
