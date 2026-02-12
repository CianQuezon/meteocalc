"""
Docstring for meteocalc.dewpoint._dewpoint_equations
"""


from abc import ABC, abstractmethod

from typing import Tuple, Union, NamedTuple, Optional, Callable
import warnings

import numpy as np
import numpy.typing as npt


from meteocalc.shared._enum_tools import parse_enum
from meteocalc.dewpoint._enums import DewPointEquationName
from meteocalc.vapor._enums import SurfaceType
from meteocalc.dewpoint._jit_equations import (
    _magnus_equation_scalar, _magnus_equation_vectorised
)
from meteocalc.dewpoint._dewpoint_constants import (
    MAGNUS_ICE, MAGNUS_WATER,
)


class DewPointEquation(ABC):
    """
    Docstring for DewPointEquation
    """
    name: DewPointEquationName
    temp_bounds: Tuple[float, float]
    surface_type: SurfaceType

    def __init__(self, surface_type: Union[str, SurfaceType]):       
        self.surface_type = parse_enum(value=surface_type, enum_class=SurfaceType)
        self._update_temp_bounds()

    def _dispatch_scalar_or_vector(self, temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike]
                                ,scalar_func: Callable[[float, float], float], vector_func: Callable[[npt.NDArray, npt.NDArray], npt.NDArray], 
                                equation_constant: Optional[NamedTuple]) -> Union[float, npt.NDArray]:
        """
        Docstring for _dispatch_scalar_or_vector
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[float, npt.ArrayLike]
        :param equation_constants: Description
        :type equation_constants: NamedTuple
        :return: Description
        :rtype: float | NDArray
        """

        temp_k, rh = self._validate_input(
            temp_k=temp_k, rh=rh
        )

        if temp_k.ndim == 0 and rh.ndim == 0:
            
            temp_k_scalar = float(temp_k.item())
            rh_scalar = float(rh.item())

            if equation_constant is not None:
                return scalar_func(temp_k_scalar, rh_scalar, *equation_constant)
            
            else: 
                return scalar_func(temp_k_scalar, rh_scalar)
        
        else:
            temp_k_original_shape = temp_k.shape

            temp_k_flatten = temp_k.flatten()
            rh_flatten = rh.flatten()

            if equation_constant is not None:
                dewpoint_temp =  vector_func(
                    temp_k_flatten, rh_flatten, *equation_constant
                )
            else:
                dewpoint_temp = vector_func(
                    temp_k_flatten, rh_flatten
                )
            return dewpoint_temp.reshape(temp_k_original_shape)


    def _broadcast_input(self, temp_k: Union[float, npt.ArrayLike], 
                         rh: Union[float, npt.ArrayLike]) -> tuple[Union[float, npt.ArrayLike], Union[float, npt.ArrayLike]]:


        if temp_k.ndim == 0 and rh.ndim == 0:
            pass

        elif temp_k.ndim == 0 and rh.ndim > 0:
            temp_k = np.full_like(rh, temp_k, dtype=np.float64)
        
        elif rh.ndim == 0 and temp_k.ndim > 0:
            rh = np.full_like(temp_k, rh, dtype=np.float64)
        
        elif temp_k.shape != rh.shape:
            raise ValueError(
                f"Input arrays must have the same shape or one must be scalar. "
                f"Got temp_k.shape ={temp_k.shape}, rh.shape={rh.shape}"
            )
        
        return temp_k, rh

    def _validate_input(self, temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike]):
        """
        Docstring for _validate_input
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[float, npt.ArrayLike]
        """
        temp_k = np.asarray(temp_k, dtype=np.float64)
        rh = np.asarray(rh, dtype=np.float64)

        temp_k, rh = self._broadcast_input(temp_k=temp_k, rh=rh)

        if np.any(rh < 0) or np.any(rh > 1):
            raise ValueError(
                f"Relative humidity must be in [0, 1] "
                f"Got range [{np.min(rh)}, {np.max(rh)}]"
            )
        
        temp_min, temp_max = self.temp_bounds

        if np.any(temp_k < temp_min) or np.any(temp_k > temp_max):
            warnings.warn(
                f"Temperature outside valid range "
                f"[{temp_min}K, {temp_max}K] "
                f"for {self.name.value} with {self.surface_type.value} surface. "
                f"Results may be inaccurate.",
                UserWarning,
                stacklevel=3           
            )
        return temp_k, rh

    @abstractmethod
    def _update_temp_bounds(self) -> None:
        pass

    @abstractmethod
    def calculate(self, temp_k: Union[float, npt.ArrayLike],
                  rh: Union[float, npt.ArrayLike]):
        """
        Docstring for calculate
        """
        pass



class MagnusDewpointEquation(DewPointEquation):
    """
    Docstring for MagnusDewpointEquation
    """
    name: DewPointEquationName = DewPointEquationName.MAGNUS

    def _update_temp_bounds(self):
        
        if self.surface_type == SurfaceType.ICE:
            self.temp_bounds = (233.15, 273.15)
        
        elif self.surface_type == SurfaceType.WATER:
            self.temp_bounds = (233.15, 333.15)
    
    def calculate(self, temp_k: Union[float, npt.ArrayLike], rh: Union[npt.ArrayLike, float]):

        if self.surface_type == SurfaceType.WATER:
            return self._dispatch_scalar_or_vector(
                temp_k=temp_k,
                rh=rh,
                scalar_func=_magnus_equation_scalar,
                vector_func=_magnus_equation_vectorised,
                equation_constant=MAGNUS_WATER
            )
        
        elif self.surface_type == SurfaceType.ICE:
            return self._dispatch_scalar_or_vector(
                temp_k=temp_k,
                rh=rh,
                scalar_func=_magnus_equation_scalar,
                vector_func=_magnus_equation_vectorised,
                equation_constant=MAGNUS_ICE
            )
        
class SolverDewpointMethod(DewPointEquation):
    """
    Docstring for SolverDewpointMethod
    """

        