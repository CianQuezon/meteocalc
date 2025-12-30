"""
This is the main py file for vapor commands for the user to use.

Commands:
Author: Cian Quezon
"""

import numpy as np

from typing import Union, List
from meteorological_equations.shared._enum_tools import parse_enum
from meteorological_equations.vapor._enums import SurfaceType, EquationName
from meteorological_equations.vapor._vapor_equations import (
    VaporEquation,
    BoltonEquation,
    GoffGratchEquation,
    HylandWexlerEquation
)

EQUATION_REGISTRY = {
    EquationName.BOLTON: BoltonEquation,
    EquationName.GOFF_GRATCH: GoffGratchEquation,
    EquationName.HYLAND_WEXLER: HylandWexlerEquation
}

class Vapor:
    def list_equations() -> List[str]:
        """
        lists and prints the available equations
        """
        return [equation.value for equation in EquationName]


    def get_equation(equation: Union[str, EquationName], phase: Union[SurfaceType, str] = SurfaceType.AUTOMATIC) -> VaporEquation:
        """
        gets the specific saturation vapor equation.

        Args:
            - equation (Union[str, EquationName]) = Equation name or enum the user requires
        
        Returns:
            Returns the equation class needed by the user

        """

        equation_enum = parse_enum(equation, EquationName)
        phase_enum = parse_enum(phase, SurfaceType)
        equation_selected = EQUATION_REGISTRY[equation_enum](surface_type = phase_enum)
        
        return equation_selected
        

    def get_vapor_saturation(temp_k: Union[np.ndarray, float], phase: SurfaceType = SurfaceType.WATER, 
                            equation: Union[EquationName, str] = EquationName.GOFF_GRATCH ) -> Union[np.ndarray, float]:
        """
        gets the vapor pressure saturation using the selected equation at a given temperature in hPa.

        Args:):
            return super().f
            - temp_k (Union[np.ndarray, float]) = a scalar or array of temperature in Kelvin.
            - phase (SurfaceType) = Phase of the saturation vapor. Phase available are "automatic", "ice", and "water"
            - equation (EquationName) = Equations used to get the vapor saturation. (i.e "bolton", "goff_gratch" etc.)   
            
        Returns:
            - scalar or an array of pressure in hPa
        """
        equation_selected = get_equation(equation, phase=phase)
        vapor_saturation = equation_selected.calculate(temp_k=temp_k)
        return vapor_saturation

