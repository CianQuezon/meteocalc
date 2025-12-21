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
from dataclasses import dataclass
from typing import Tuple, ClassVar, Union
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

    def _detect_surface_type(self, temp_k: Union[float, npt.ArrayLike], ice_min_range: float, ice_max_range: float, 
                             water_min_range: float, water_max_range: float) -> SurfaceType:
        """
        automatic surface type detection based on over water and ice temperature range.
        
        Args:
            - ice_min_range (float) = minimum temp range for over ice constants in Kelvin
            - ice_max_range (float) = maximum temp range for over ice constants in Kelvin
            - water_min_range (float) = minimum temp range for over water constants in Kelvin
            - water_max_range (float) = maximum temp range for over water constants in Kelvin
        
        Returns:
            - Detected Surface type        
        """

        if 



    @abstractmethod
    def calculate(self, temp_k: Union[npt.ArrayLike, float]) -> Union[npt.NDArray, np.float64]:
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
    """
    surface_type: SurfaceType = SurfaceType.WATER
    temp_bounds: Tuple[float, float] = (238.15, 308.15)
    name: EquationName = EquationName.BOLTON

    def calculate(self, temp_k: Union[npt.ArrayLike, float]) -> Union[npt.NDArray, np.float64]:
        
        self._check_bounds(temp_k=temp_k)
        
        temp_k = np.asarray(temp_k, dtype=np.float64)
        original_shape = temp_k.shape

        if temp_k.ndim == 0:
            return _bolton_scalar(temp_k=temp_k.item())
        else:
            temp_k_flat = temp_k.flatten()
            result = _bolton_vectorised(temp_k_flat)
            return result.reshape(original_shape)

class GoffGratchEquation(VaporEquation):
    """
    Calculates saturation vapor using Goff Gratch Equation
    """
    surface_type: SurfaceType = SurfaceType.AUTOMATIC
    temp_bounds: Tuple[float, float] = (173.15, 373.15)
    name: EquationName = EquationName.GOFF_GRATCH


