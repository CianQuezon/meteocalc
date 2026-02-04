"""
An equation interface for using the jit and constants.

Implements:
- Bolton
- Goff Gratch
- Hyland Wexler

Author: Cian Quezon
"""

import warnings
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from meteorological_equations.vapor._enums import EquationName, SurfaceType
from meteorological_equations.vapor._jit_equations import (
    _bolton_scalar,
    _bolton_vectorised,
    _goff_gratch_scalar,
    _goff_gratch_vector,
    _hyland_wexler_scalar,
    _hyland_wexler_vectorised,
)
from meteorological_equations.vapor._vapor_constants import (
    GOFF_GRATCH_ICE,
    GOFF_GRATCH_WATER,
    HYLAND_WEXLER_ICE,
    HYLAND_WEXLER_WATER,
)


class VaporEquation(ABC):
    """Abstract base class for vapor pressure equations.

    This class provides a common interface for different saturation vapor pressure
    calculation methods. Subclasses implement specific equations (Bolton, Goff-Gratch,
    Hyland-Wexler) with support for calculations over water, ice, or automatic surface
    type detection.

    Parameters
    ----------
    surface_type : SurfaceType or str, default=SurfaceType.AUTOMATIC
        Surface type for vapor pressure calculations. Options are:
        - 'automatic': Auto-selects ice/water based on temperature
        - 'ice': Uses ice surface constants
        - 'water': Uses water surface constants

    Attributes
    ----------
    surface_type : SurfaceType
        The surface type used for calculations.
    temp_bounds : tuple of float
        Valid temperature range (min, max) in Kelvin for the equation.
    name : EquationName
        Identifier for the equation type.

    Raises
    ------
    ValueError
        If an invalid surface type string is provided.
    """

    surface_type: SurfaceType
    temp_bounds: Tuple[float, float]
    name: EquationName

    def __init__(self, surface_type: Union[SurfaceType, str] = SurfaceType.AUTOMATIC):
        if isinstance(surface_type, str):
            try:
                surface_type = SurfaceType(surface_type.lower())
            except ValueError as err:
                valid = [s.value for s in SurfaceType]
                raise ValueError(
                    f"Invalid surface type: '{surface_type}'Valid ones are: {valid}"
                ) from err
        self.surface_type = surface_type
        self._update_temp_bounds()

    def _calculate_automatic_equation(
        self,
        temp_k: Union[np.ndarray, float],
        scalar_func: Callable[[float, Optional[NamedTuple]], float],
        vector_func: Callable[[np.ndarray, Optional[NamedTuple]], np.ndarray],
        water_constants: NamedTuple,
        ice_constants: NamedTuple,
    ) -> Union[np.ndarray, float]:
        """Automatically select and calculate vapor pressure based on temperature.

        Determines surface type based on temperature relative to freezing point:
        - Above 273.15 K: uses water constants
        - Below 273.15 K: uses ice constants
        - At 273.15 K (within tolerance): averages water and ice results

        Parameters
        ----------
        temp_k : float or array_like
            Temperature(s) in Kelvin.
        scalar_func : callable
            Scalar calculation function.
        vector_func : callable
            Vectorized calculation function.
        water_constants : NamedTuple
            Constants for water surface calculations.
        ice_constants : NamedTuple
            Constants for ice surface calculations.

        Returns
        -------
        float or ndarray
            Saturation vapor pressure in hPa.
        """

        temp_arr = np.asarray(temp_k)

        at_freezing, above_freezing, below_freezing = self._detect_surface_type(
            temp_k=temp_arr
        )

        if temp_arr.ndim == 0:
            temp_val = temp_arr.item()

            if above_freezing.item():
                return scalar_func(temp_val, *water_constants)

            if below_freezing.item():
                return scalar_func(temp_val, *ice_constants)

            on_water_vapor = scalar_func(temp_val, *water_constants)
            on_ice_vapor = scalar_func(temp_val, *ice_constants)
            average = (on_water_vapor + on_ice_vapor) / 2
            return average

        else:
            result = np.empty_like(temp_arr, dtype=np.float64)

            if np.any(above_freezing):
                result[above_freezing] = vector_func(
                    temp_arr[above_freezing], *water_constants
                )
            if np.any(below_freezing):
                result[below_freezing] = vector_func(
                    temp_arr[below_freezing], *ice_constants
                )
            if np.any(at_freezing):
                over_ice_results = vector_func(temp_arr[at_freezing], *ice_constants)
                over_water_results = vector_func(
                    temp_arr[at_freezing], *water_constants
                )
                result[at_freezing] = (over_ice_results + over_water_results) / 2
            return cast(np.ndarray, result)

    def _dispatch_scalar_or_vector(
        self,
        temp_k: Union[npt.ArrayLike, float],
        scalar_func: Callable[[float], float],
        vector_func: Callable[[np.ndarray], np.ndarray],
        equation_constant: Optional[NamedTuple],
    ) -> Union[np.ndarray, float]:
        """Dispatch to scalar or vectorized calculation based on input shape.

        Parameters
        ----------
        temp_k : float or array_like
            Temperature(s) in Kelvin.
        scalar_func : callable
            Function for scalar calculations.
        vector_func : callable
            Function for vectorized calculations.
        equation_constant : NamedTuple or None
            Equation constants (water or ice).

        Returns
        -------
        float or ndarray
            Saturation vapor pressure in hPa.
        """
        temp_k = np.asarray(temp_k)
        if temp_k.ndim == 0:
            if equation_constant is not None:
                return scalar_func(temp_k.item(), *equation_constant)
            else:
                return scalar_func(temp_k.item())

        else:
            temp_k_original_shape = temp_k.shape
            temp_k_flatten = temp_k.flatten()

            if equation_constant is not None:
                vapor_pressure = vector_func(temp_k_flatten, *equation_constant)
            else:
                vapor_pressure = vector_func(temp_k_flatten)
            return cast(np.ndarray, vapor_pressure.reshape(temp_k_original_shape))

    def _check_bounds(self, temp_k: Union[npt.ArrayLike, float]) -> None:
        """Check if temperature is within valid equation range.

        Issues a warning if any temperature values fall outside the valid range.

        Parameters
        ----------
        temp_k : float or array_like
            Temperature(s) in Kelvin.

        Warnings
        --------
        UserWarning
            If temperature is outside the valid range for this equation.
        """
        temp_array = np.asarray(temp_k)
        min_bound, max_bound = self.temp_bounds

        if np.any(temp_array < min_bound) or np.any(temp_array > max_bound):
            warnings.warn(
                f"Temperature is outside the valid range [min: {min_bound}K, max: {max_bound}K]"
                f"for {self.name} equation. Results maybe inaccurate.",
                UserWarning,
                stacklevel=3,
            )

    def _detect_surface_type(
        self, temp_k: Union[float, npt.ArrayLike], tolerance: float = 1e-6
    ) -> npt.NDArray:
        """Detect surface type based on temperature relative to freezing point.

        Parameters
        ----------
        temp_k : float or array_like
            Temperature(s) in Kelvin.
        tolerance : float, default=1e-6
            Temperature tolerance for freezing point detection.

        Returns
        -------
        at_freezing : ndarray
            Boolean mask where temperature is at freezing point (within tolerance).
        above_freezing : ndarray
            Boolean mask where temperature is above freezing point.
        below_freezing : ndarray
            Boolean mask where temperature is below freezing point.
        """
        temp_k = np.asarray(temp_k)
        freezing_point = 273.15

        at_freezing = np.abs(temp_k - freezing_point) < tolerance
        above_freezing = temp_k > (freezing_point + tolerance)
        below_freezing = temp_k < (freezing_point - tolerance)

        return at_freezing, above_freezing, below_freezing

    @abstractmethod
    def _update_temp_bounds(self) -> None:
        """Update temperature bounds based on surface type.

        Must be implemented by subclasses to set appropriate temperature
        ranges for the specific equation.
        """
        pass

    @abstractmethod
    def calculate(
        self, temp_k: Union[npt.ArrayLike, float]
    ) -> Union[npt.NDArray, float]:
        """Calculate saturation vapor pressure at given temperature(s).

        Parameters
        ----------
        temp_k : float or array_like
            Temperature(s) in Kelvin.

        Returns
        -------
        float or ndarray
            Saturation vapor pressure in hPa.

        Warnings
        --------
        UserWarning
            If temperature is outside the valid range for this equation.
        """
        pass


class BoltonEquation(VaporEquation):
    """Saturation vapor pressure using Bolton's equation.

    Bolton's equation is a simplified empirical formula valid for temperatures
    between -30°C and 40°C. This equation only supports calculations over water.

    Parameters
    ----------
    surface_type : SurfaceType or str, default=SurfaceType.WATER
        Must be 'water'. Other surface types are not supported.

    Attributes
    ----------
    name : EquationName
        Set to EquationName.BOLTON.
    temp_bounds : tuple of float
        Valid temperature range: (243.15 K, 313.15 K) or (-30°C, 40°C).

    Raises
    ------
    ValueError
        If surface_type is set to 'ice' (not supported by Bolton equation).

    Examples
    --------
    >>> bolton = BoltonEquation()
    >>> temp = 293.15  # 20°C
    >>> vapor_pressure = bolton.calculate(temp)

    Notes
    -----
    Bolton's equation is: es = 6.112 * exp((17.67 * T_c) / (T_c + 243.5))
    where T_c is temperature in Celsius and es is in hPa.
    """


    name: EquationName = EquationName.BOLTON

    def __init__(self, surface_type: SurfaceType = SurfaceType.WATER):
        if isinstance(surface_type, str):
            surface_type = SurfaceType(surface_type.lower())

        if surface_type == SurfaceType.ICE:
            raise ValueError("Bolton only supports water")

        super().__init__(surface_type)

    def _update_temp_bounds(self) -> None:
        """Set Bolton equation temperature bounds (243.15 K to 313.15 K)."""
        self.temp_bounds = (243.15, 313.15)

    def calculate(
        self, temp_k: Union[npt.ArrayLike, float]
    ) -> Union[npt.NDArray, np.float64]:
        temp_k = np.asarray(temp_k)
        self._check_bounds(temp_k=temp_k)
        """Calculate saturation vapor pressure using Bolton's equation.

        Parameters
        ----------
        temp_k : float or array_like
            Temperature(s) in Kelvin.

        Returns
        -------
        float or ndarray
            Saturation vapor pressure in hPa.

        Warnings
        --------
        UserWarning
            If temperature is outside the valid range (243.15 K to 313.15 K).
        """
        return self._dispatch_scalar_or_vector(
            temp_k=temp_k,
            scalar_func=_bolton_scalar,
            vector_func=_bolton_vectorised,
            equation_constant=None,
        )


class GoffGratchEquation(VaporEquation):
    """Saturation vapor pressure using Goff-Gratch equation.

    The Goff-Gratch equation is a highly accurate formulation recommended by
    the World Meteorological Organization (WMO). It supports calculations over
    both water and ice surfaces.

    Parameters
    ----------
    surface_type : SurfaceType or str, default=SurfaceType.AUTOMATIC
        Surface type for calculations:
        - 'automatic': Auto-selects based on temperature
        - 'water': Over liquid water (273.15 K to 373.15 K)
        - 'ice': Over ice (173.15 K to 273.16 K)

    Attributes
    ----------
    name : EquationName
        Set to EquationName.GOFF_GRATCH.
    temp_bounds : tuple of float
        Valid temperature range depending on surface_type:
        - automatic: (173.15 K, 373.15 K)
        - water: (273.15 K, 373.15 K)
        - ice: (173.15 K, 273.16 K)

    Examples
    --------
    >>> # Automatic surface type selection
    >>> goff = GoffGratchEquation(surface_type='automatic')
    >>> temps = np.array([263.15, 273.15, 283.15])  # Below, at, above freezing
    >>> vapor_pressures = goff.calculate(temps)

    >>> # Explicit water surface
    >>> goff_water = GoffGratchEquation(surface_type='water')
    >>> vapor_pressure = goff_water.calculate(293.15)
    """

    name: EquationName = EquationName.GOFF_GRATCH

    def _update_temp_bounds(self) -> None:
        if self.surface_type == SurfaceType.AUTOMATIC:
            self.temp_bounds = (173.15, 373.15)
        elif self.surface_type == SurfaceType.WATER:
            self.temp_bounds = (273.15, 373.15)
        elif self.surface_type == SurfaceType.ICE:
            self.temp_bounds = (173.15, 273.16)

    def calculate(
        self, temp_k: Union[npt.ArrayLike, float]
    ) -> Union[npt.NDArray, np.float64]:
        temp_k = np.asarray(temp_k)
        self._check_bounds(temp_k=temp_k)

        if self.surface_type == SurfaceType.AUTOMATIC:
            return self._calculate_automatic_equation(
                temp_k=temp_k,
                scalar_func=_goff_gratch_scalar,
                vector_func=_goff_gratch_vector,
                water_constants=GOFF_GRATCH_WATER,
                ice_constants=GOFF_GRATCH_ICE,
            )

        elif self.surface_type == SurfaceType.WATER:
            return self._dispatch_scalar_or_vector(
                temp_k=temp_k,
                scalar_func=_goff_gratch_scalar,
                vector_func=_goff_gratch_vector,
                equation_constant=GOFF_GRATCH_WATER,
            )

        elif self.surface_type == SurfaceType.ICE:
            return self._dispatch_scalar_or_vector(
                temp_k=temp_k,
                scalar_func=_goff_gratch_scalar,
                vector_func=_goff_gratch_vector,
                equation_constant=GOFF_GRATCH_ICE,
            )


class HylandWexlerEquation(VaporEquation):
    """Saturation vapor pressure using Hyland-Wexler equation.

    The Hyland-Wexler equation provides high accuracy over a wide temperature
    range and supports calculations over both water and ice surfaces. This
    formulation is commonly used in HVAC and meteorological applications.

    Parameters
    ----------
    surface_type : SurfaceType or str, default=SurfaceType.AUTOMATIC
        Surface type for calculations:
        - 'automatic': Auto-selects based on temperature
        - 'water': Over liquid water (273.15 K to 473.15 K)
        - 'ice': Over ice (173.15 K to 273.16 K)

    Attributes
    ----------
    name : EquationName
        Set to EquationName.HYLAND_WEXLER.
    temp_bounds : tuple of float
        Valid temperature range depending on surface_type:
        - automatic: (173.15 K, 473.15 K)
        - water: (273.15 K, 473.15 K)
        - ice: (173.15 K, 273.16 K)

    Examples
    --------
    >>> # Automatic surface type selection
    >>> hyland = HylandWexlerEquation(surface_type='automatic')
    >>> temp = 298.15  # 25°C
    >>> vapor_pressure = hyland.calculate(temp)

    >>> # Calculate for array of temperatures
    >>> temps = np.linspace(273.15, 373.15, 10)
    >>> vapor_pressures = hyland.calculate(temps)
    """


    name: EquationName = EquationName.HYLAND_WEXLER

    def _update_temp_bounds(self) -> None:
        if self.surface_type == SurfaceType.AUTOMATIC:
            self.temp_bounds = (173.15, 473.15)
        elif self.surface_type == SurfaceType.WATER:
            self.temp_bounds = (273.15, 473.15)
        elif self.surface_type == SurfaceType.ICE:
            self.temp_bounds = (173.15, 273.16)

    def calculate(
        self, temp_k: Union[npt.ArrayLike, float]
    ) -> Union[npt.NDArray, np.float64]:
        temp_k = np.asarray(temp_k)
        self._check_bounds(temp_k=temp_k)

        if self.surface_type == SurfaceType.AUTOMATIC:
            return self._calculate_automatic_equation(
                temp_k=temp_k,
                scalar_func=_hyland_wexler_scalar,
                vector_func=_hyland_wexler_vectorised,
                water_constants=HYLAND_WEXLER_WATER,
                ice_constants=HYLAND_WEXLER_ICE,
            )

        elif self.surface_type == SurfaceType.WATER:
            return self._dispatch_scalar_or_vector(
                temp_k=temp_k,
                scalar_func=_hyland_wexler_scalar,
                vector_func=_hyland_wexler_vectorised,
                equation_constant=HYLAND_WEXLER_WATER,
            )

        elif self.surface_type == SurfaceType.ICE:
            return self._dispatch_scalar_or_vector(
                temp_k=temp_k,
                scalar_func=_hyland_wexler_scalar,
                vector_func=_hyland_wexler_vectorised,
                equation_constant=HYLAND_WEXLER_ICE,
            )
