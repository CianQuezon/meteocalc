"""
An equation interface for using the jit and constants.

Implements:
- Bolton
- Goff Gratch
- Hyland Wexler

Author: Cian Quezon
"""
import warnings
import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import Tuple, Union, Callable, NamedTuple, Optional, cast
from meteorological_equations.vapor._enums import SurfaceType, EquationName
from meteorological_equations.vapor._vapor_constants import (
    GOFF_GRATCH_ICE, GOFF_GRATCH_WATER, 
    HYLAND_WEXLER_ICE, HYLAND_WEXLER_WATER
)
from meteorological_equations.vapor._jit_equations import (
    _bolton_scalar, _bolton_vectorised,
    _goff_gratch_scalar, _goff_gratch_vector,
    _hyland_wexler_scalar, _hyland_wexler_vectorised
)

class VaporEquation(ABC):
    """
    Abstract Base class for vapor pressure equations.

    Attributes:
        - over_ice (bool) = vapor pressure over liquid or ice
        - temp_bounds(Tuple[float, float]) = Temperature range (min, max) in kelvin for the equation 
        - name (str) = Class variable for identifying the equation name.
    """
    surface_type: SurfaceType 
    temp_bounds: Tuple[float, float]
    name: EquationName

    def __init__(self, surface_type: Union[SurfaceType, str] = SurfaceType.AUTOMATIC):
        if isinstance(surface_type, str):
            try:
                surface_type = SurfaceType(surface_type.lower())
            except ValueError:
                valid = [s.value for s in SurfaceType]
                raise ValueError(
                    f"Invalid surface type: '{surface_type}'"
                    f"Valid ones are: {valid}"
                )
        self.surface_type = surface_type
        self._update_temp_bounds()

    def _calculate_automatic_equation(self, temp_k: Union[np.ndarray, float], scalar_func: Callable[[float, Optional[NamedTuple]], float],
                                      vector_func: Callable[[np.ndarray, Optional[NamedTuple]], np.ndarray], water_constants: NamedTuple, ice_constants: NamedTuple) -> Union[np.ndarray, float]:
        """
        auto detects the temperature at the following classifications:
            - above_freezing = above freezing point (above (273.15 + tolerance)). It uses over water constants
            - below_freezing = below freezing point (below (273.15 + tolerance)). It uses over ice constants
            - at_freezing = at freezing point ((abs(temp - 273.15)) <= tolerance). It averages both water and ice constant results.

        Args:
            - temp_k (Union[npt.ArrayLike, float]) = scalar or an array of temperatures in Kelvin
            - scalar_func(Callable[[float, Optional[NamedTuple]], float]) = scalar function that takes in a temperature with optional, either water or ice constants.
            - vector_func(Callable[[np.ndarray, Optional[NamedTuple]], np.ndarray]) = vectorised function that takes in temperature array and outputs an array of vapor saturation in hPa. Optionally uses water or ice constants.
            - water_constants(Optional[Named_Tuple]) = water constants for over water equation
            - ice_constants(Optional[Named_Tuple]) =ice constants for over ice equation
        
        Returns:
            Array or scalar saturation vapor in hPA
        """
        
        temp_arr = np.asarray(temp_k)

        at_freezing, above_freezing, below_freezing = self._detect_surface_type(temp_k=temp_arr)

        if temp_arr.ndim == 0:

            temp_val = temp_arr.item()
 
            if above_freezing.item():
                return scalar_func(temp_val, *water_constants)
            
            if below_freezing.item():
                return scalar_func(temp_val, *ice_constants)

            on_water_vapor = scalar_func(temp_val, *water_constants)
            on_ice_vapor = scalar_func(temp_val, *ice_constants)
            average = (on_water_vapor + on_ice_vapor)/2
            return average
        
        else:
            result = np.empty_like(temp_arr, dtype=np.float64)

            if np.any (above_freezing):
                result[above_freezing] = vector_func(temp_arr[above_freezing], *water_constants)
            if np.any (below_freezing):
                result[below_freezing] = vector_func(temp_arr[below_freezing], *ice_constants)
            if np.any(at_freezing):
                over_ice_results = vector_func(temp_arr[at_freezing], *ice_constants)
                over_water_results = vector_func(temp_arr[at_freezing], *water_constants)
                result[at_freezing] = (over_ice_results + over_water_results)/2
            return cast(np.ndarray, result)

    def _dispatch_scalar_or_vector(self, temp_k: Union[npt.ArrayLike, float] , scalar_func: Callable[[float], float], 
                                   vector_func: Callable[[np.ndarray], np.ndarray], equation_constant: Optional[NamedTuple]) -> Union[np.ndarray, float]:
        """
        calculates saturation vapor using the equation's scalar or vectorised function depending on
        input temperature dimensions.  
        
        Args:
         - temp_k (Union[npt.ArrayLike, float]) = scalar or an array of temperatures in Kelvin
         - scalar_func(Callable[[float, Optional[NamedTuple]], float]) = scalar function that takes in a temperature with optional, either water or ice constants.
         - vector_func(Callable[[np.ndarray, Optional[NamedTuple]], np.ndarray]) = vectorised function that takes in temperature array and outputs an array of vapor saturation in hPa. Optionally uses water or ice constants.
         - equation_constants(Optional[Named_Tuple]) = Optionally passes either ice or water constants for the vapor equation
         
         Returns:
          Array or scalar saturation vapor in hPA
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
        """
        Checks if temperature is within the valid range and sets out a warning if not.
        
        Args:
         - temp_k (Union[npt.ArrayLike, float]) = Array or scalar temperature in kelvin 
        """
        temp_array = np.asarray(temp_k)
        min_bound, max_bound = self.temp_bounds

        if np.any(temp_array < min_bound) or np.any(temp_array > max_bound):
            warnings.warn(
                f"Temperature is outside the valid range [min: {min_bound}K, max: {max_bound}K]"
                f"for {self.name} equation. Results maybe inaccurate.",
                UserWarning,
                stacklevel=3
            )

    def _detect_surface_type(self, temp_k: Union[float, npt.ArrayLike], tolerance: float = 1e-6) -> npt.NDArray:
        """
        automatic surface type detection based on over water and ice temperature range.
        
        Args:
            - temp_k (Union[float, npt.ArrayLike]) = Scalar or array of temperature in Kelvin.
            - tolerance (float) = tolerance of temperature considered to be at freezing point.
        Returns:
            - at_freezing (npt.NDArray) = Boolean mask of temperature at freezing point and within tolerance
            - above_freezing (npt.NDArray) = Boolean mask of temperature above freezing point        
            - below_freezing (npt.NDArray) = Boolean mask of temperature below freezing point
        """
        temp_k = np.asarray(temp_k)
        freezing_point = 273.15

        at_freezing = np.abs(temp_k - freezing_point) < tolerance
        above_freezing = temp_k > (freezing_point + tolerance)
        below_freezing = temp_k < (freezing_point - tolerance)
        
        return at_freezing, above_freezing, below_freezing

    @abstractmethod
    def _update_temp_bounds(self) -> None:
        """
        Sets the temperature bounds of the equation
        """
        pass
    
    @abstractmethod
    def calculate(self, temp_k: Union[npt.ArrayLike, float]) -> Union[npt.NDArray, float]:
        """
        Calculates saturation vapor at a given temperature.

        Args:
         - temp_k (Union[npt.ArrayLike, float]) = Array or scalar temperature in kelvin 
        
        Returns:
            Returns either an array [npt.ArrayLike] or scalar [np.float64] of saturated vapour in hPa
        """
        pass

class BoltonEquation(VaporEquation):
    """
    Calculates saturation vapor using Bolton Equation

    Notes:
        - Bolton Equation has only over water calculations
    """
    name: EquationName = EquationName.BOLTON

    def __init__(self, surface_type: SurfaceType = SurfaceType.WATER):
        
        if isinstance(surface_type, str):
            surface_type = SurfaceType(surface_type.lower())
        
        if surface_type == SurfaceType.ICE:
            raise ValueError("Bolton only supports water")
        
        super().__init__(surface_type)

    def _update_temp_bounds(self) -> None:
        """
        Bolton has a fixed temperature range since it only supports over water equation.
        """
        self.temp_bounds = (243.15, 313.15)

    def calculate(self, temp_k: Union[npt.ArrayLike, float]) -> Union[npt.NDArray, np.float64]:
        temp_k = np.asarray(temp_k)    
        self._check_bounds(temp_k=temp_k)
        
        return self._dispatch_scalar_or_vector(temp_k=temp_k, scalar_func=_bolton_scalar, vector_func=_bolton_vectorised,
                                        equation_constant=None)

class GoffGratchEquation(VaporEquation):
    """
    Calculates saturation vapor using Goff Gratch Equation
    """
    name: EquationName = EquationName.GOFF_GRATCH

    def _update_temp_bounds(self) -> None:
        if self.surface_type == SurfaceType.AUTOMATIC:
            self.temp_bounds = (173.15, 373.15)
        elif self.surface_type == SurfaceType.WATER:
            self.temp_bounds = (273.15, 373.15)
        elif self.surface_type == SurfaceType.ICE:
            self.temp_bounds = (173.15, 273.16)
    
    def calculate(self, temp_k: Union[npt.ArrayLike, float]) -> Union[npt.NDArray, np.float64]:

        temp_k = np.asarray(temp_k)
        self._check_bounds(temp_k=temp_k)

        if self.surface_type == SurfaceType.AUTOMATIC:
            return self._calculate_automatic_equation(temp_k=temp_k, scalar_func=_goff_gratch_scalar,
                                                      vector_func=_goff_gratch_vector, water_constants=GOFF_GRATCH_WATER, ice_constants=GOFF_GRATCH_ICE)
        
        elif self.surface_type == SurfaceType.WATER:
            return self._dispatch_scalar_or_vector(temp_k=temp_k, scalar_func=_goff_gratch_scalar,
                                            vector_func=_goff_gratch_vector, equation_constant=GOFF_GRATCH_WATER)

        elif self.surface_type == SurfaceType.ICE:
            return self._dispatch_scalar_or_vector(temp_k=temp_k, scalar_func=_goff_gratch_scalar,
                                            vector_func=_goff_gratch_vector, equation_constant=GOFF_GRATCH_ICE)

class HylandWexlerEquation(VaporEquation):
    """
    Calculates saturation vapor using Hyland Wexler Equation
    """
    name: EquationName = EquationName.HYLAND_WEXLER

    def _update_temp_bounds(self) -> None:
        
        if self.surface_type == SurfaceType.AUTOMATIC:
            self.temp_bounds = (173.15, 473.15)
        elif self.surface_type == SurfaceType.WATER:
            self.temp_bounds = (273.15, 473.15)
        elif self.surface_type == SurfaceType.ICE:
            self.temp_bounds = (173.15, 273.16)

    def calculate(self, temp_k: Union[npt.ArrayLike, float]) -> Union[npt.NDArray, np.float64]:
        temp_k = np.asarray(temp_k)
        self._check_bounds(temp_k=temp_k)

        if self.surface_type == SurfaceType.AUTOMATIC:
            return self._calculate_automatic_equation(temp_k=temp_k, scalar_func=_hyland_wexler_scalar,
                                                      vector_func=_hyland_wexler_vectorised, water_constants=HYLAND_WEXLER_WATER, ice_constants=HYLAND_WEXLER_ICE)
        
        elif self.surface_type == SurfaceType.WATER:
            return self._dispatch_scalar_or_vector(temp_k=temp_k, scalar_func=_hyland_wexler_scalar,
                                            vector_func=_hyland_wexler_vectorised, equation_constant=HYLAND_WEXLER_WATER)
        
        elif self.surface_type == SurfaceType.ICE:
            return self._dispatch_scalar_or_vector(temp_k=temp_k, scalar_func=_hyland_wexler_scalar,
                                            vector_func=_hyland_wexler_vectorised, equation_constant=HYLAND_WEXLER_ICE)


