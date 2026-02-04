"""
This is the main py file for vapor commands for the user to use.

Commands:
Author: Cian Quezon
"""

from typing import List, Union

import numpy as np

from meteorological_equations.shared._enum_tools import parse_enum
from meteorological_equations.vapor._enums import EquationName, SurfaceType
from meteorological_equations.vapor._vapor_equations import (
    BoltonEquation,
    GoffGratchEquation,
    HylandWexlerEquation,
    VaporEquation,
)

EQUATION_REGISTRY = {
    EquationName.BOLTON: BoltonEquation,
    EquationName.GOFF_GRATCH: GoffGratchEquation,
    EquationName.HYLAND_WEXLER: HylandWexlerEquation,
}


class Vapor:
    """User interface for saturation vapor pressure calculations.
    
    This class provides static methods for listing available equations,
    selecting specific equation implementations, and calculating saturation
    vapor pressure.
    
    Examples
    --------
    >>> # List available equations
    >>> Vapor.list_equations()
    ['bolton', 'goff_gratch', 'hyland_wexler']
    
    >>> # Quick calculation using default (Goff-Gratch)
    >>> temp = 293.15  # 20°C in Kelvin
    >>> vapor_pressure = Vapor.get_vapor_saturation(temp)
    
    >>> # Calculate with specific equation and surface type
    >>> temps = np.array([263.15, 273.15, 283.15])
    >>> vapor_pressures = Vapor.get_vapor_saturation(
    ...     temps, 
    ...     equation='hyland_wexler',
    ...     phase='water'
    ... )
    
    >>> # Get equation instance for repeated calculations
    >>> bolton = Vapor.get_equation('bolton')
    >>> result1 = bolton.calculate(293.15)
    >>> result2 = bolton.calculate(298.15)
    """
    @staticmethod
    def list_equations() -> List[str]:
        """List all available saturation vapor pressure equations.
        
        Returns
        -------
        list of str
            Names of available equations: 'bolton', 'goff_gratch', 'hyland_wexler'.
        
        Examples
        --------
        >>> equations = Vapor.list_equations()
        >>> print(equations)
        ['bolton', 'goff_gratch', 'hyland_wexler']
        """
        return [equation.value for equation in EquationName]

    @staticmethod
    def get_equation(
        equation: Union[str, EquationName],
        phase: Union[SurfaceType, str] = SurfaceType.AUTOMATIC,
    ) -> VaporEquation:
        """Get a specific saturation vapor pressure equation instance.
        
        Useful when you need to perform multiple calculations with the same
        equation configuration.
        
        Parameters
        ----------
        equation : str or EquationName
            Name of the equation to use. Options:
            - 'bolton': Bolton's equation (water only, -30°C to 40°C)
            - 'goff_gratch': Goff-Gratch equation (WMO standard)
            - 'hyland_wexler': Hyland-Wexler equation (wide temperature range)
        phase : SurfaceType or str, default='automatic'
            Surface type for calculations:
            - 'automatic': Auto-selects ice/water based on temperature
            - 'water': Over liquid water
            - 'ice': Over ice
        
        Returns
        -------
        VaporEquation
            Instance of the requested equation class.
        
        Raises
        ------
        ValueError
            If equation name is invalid or incompatible phase is requested
            (e.g., 'ice' with Bolton equation).
        
        Examples
        --------
        >>> # Get Goff-Gratch equation with automatic phase detection
        >>> goff = Vapor.get_equation('goff_gratch', phase='automatic')
        >>> vapor_pressure = goff.calculate(273.15)
        
        >>> # Get Bolton equation (only supports water)
        >>> bolton = Vapor.get_equation('bolton')
        >>> result = bolton.calculate(293.15)
        """

        equation_enum = parse_enum(equation, EquationName)
        phase_enum = parse_enum(phase, SurfaceType)
        equation_selected = EQUATION_REGISTRY[equation_enum](surface_type=phase_enum)

        return equation_selected

    @staticmethod
    def get_vapor_saturation(
        temp_k: Union[np.ndarray, float],
        phase: SurfaceType = SurfaceType.AUTOMATIC,
        equation: Union[EquationName, str] = EquationName.GOFF_GRATCH,
    ) -> Union[np.ndarray, float]:
        """Calculate saturation vapor pressure at given temperature(s).
        
        This is the primary method for calculating saturation vapor pressure.
        By default, it uses the Goff-Gratch equation with automatic surface
        type detection.
        
        Parameters
        ----------
        temp_k : float or array_like
            Temperature(s) in Kelvin. Can be a scalar or numpy array.
        phase : SurfaceType or str, default='automatic'
            Surface type for calculations:
            - 'automatic': Auto-selects ice/water based on temperature
            - 'water': Over liquid water surface
            - 'ice': Over ice surface
        equation : EquationName or str, default='goff_gratch'
            Equation to use for calculation:
            - 'bolton': Fast, simple (water only, limited range)
            - 'goff_gratch': WMO standard, high accuracy
            - 'hyland_wexler': Wide temperature range
        
        Returns
        -------
        float or ndarray
            Saturation vapor pressure in hectoPascals (hPa). Returns same
            shape as input temperature.
        
        Raises
        ------
        ValueError
            If invalid equation or phase is specified.
        
        Warnings
        --------
        UserWarning
            If temperature is outside the valid range for the selected equation.
        
        Examples
        --------
        >>> # Single temperature calculation
        >>> temp = 293.15  # 20°C
        >>> es = Vapor.get_vapor_saturation(temp)
        
        >>> # Array of temperatures with automatic phase detection
        >>> temps = np.array([263.15, 273.15, 283.15])  # -10°C, 0°C, 10°C
        >>> es_array = Vapor.get_vapor_saturation(temps)
        
        >>> # Specify equation and phase explicitly
        >>> es = Vapor.get_vapor_saturation(
        ...     temp_k=298.15,
        ...     equation='hyland_wexler',
        ...     phase='water'
        ... )
        
        >>> # Calculate over ice surface only
        >>> temp_ice = 263.15  # -10°C
        >>> es_ice = Vapor.get_vapor_saturation(
        ...     temp_ice,
        ...     phase='ice',
        ...     equation='goff_gratch'
        ... )
        
        Notes
        -----
        The Goff-Gratch equation is recommended for most meteorological
        applications as it is the WMO standard and provides high accuracy
        over a wide temperature range.
        """
        equation_selected = Vapor.get_equation(equation, phase=phase)
        vapor_saturation = equation_selected.calculate(temp_k=temp_k)
        return vapor_saturation
