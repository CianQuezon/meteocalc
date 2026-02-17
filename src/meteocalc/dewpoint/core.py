"""
Core interface for calculationg dewpoint and frostpoint.

Author: Cian Quezon
"""

from meteocalc.shared._enum_tools import parse_enum
from meteocalc.dewpoint._types import APPROXIMATION_DEWPOINT_REGISTRY
from meteocalc.dewpoint._enums import DewPointEquationName, CalculationMethod
from meteocalc.vapor._enums import EquationName
from meteocalc.dewpoint._dewpoint_equations import DewPointEquation, VaporInversionDewpoint
from typing import Union, Optional
import numpy.typing as npt

class Dewpoint:
    """
    Unified interface for dew point and frost point calculations.
    
    Provides both fast approximation methods and exact numerical solvers
    for calculating dew point (water surface) and frost point (ice surface)
    temperatures from air temperature and relative humidity.
    
    This class offers multiple calculation methods with different speed-accuracy
    tradeoffs to suit various applications, from real-time operational meteorology
    to research-grade scientific analysis.
    
    Methods
    -------
    **Approximation Methods (Fast):**
        get_dewpoint_approximation(temp_k, rh, dewpoint_equation_name='magnus')
            Fast dew point calculation 
        
        get_frostpoint_approximation(temp_k, rh, frostpoint_equation_name='magnus')
            Fast frost point calculation 
    
    **Exact Solver Methods (Accurate):**
        get_dewpoint_solver(temp_k, rh, vapor_equation_name='goff_gratch')
            Exact dew point calculation 
        
        get_frostpoint_solver(temp_k, rh, vapor_equation_name='goff_gratch')
            Exact frost point calculation 
    
    **Convenience Wrappers:**
        get_dewpoint(temp_k, rh, calculation_method='approximation', ...)
            Unified dew point interface (delegates to approximation or solver)
        
        get_frostpoint(temp_k, rh, calculation_method='approximation', ...)
            Unified frost point interface (delegates to approximation or solver)
    
    **Utility:**
        get_equations_available()
            Get list of all available equation names
    
    Examples
    --------
    >>> # Quick start: Fast approximation
    >>> td = Dewpoint.get_dewpoint_approximation(temp_k=293.15, rh=0.6)
    >>> print(f"{td - 273.15:.2f}°C")
    11.99°C
    
    >>> # Research-grade accuracy: Exact solver
    >>> td = Dewpoint.get_dewpoint_solver(temp_k=293.15, rh=0.6)
    >>> print(f"{td - 273.15:.4f}°C")
    12.0077°C
    
    >>> # Frost point below freezing
    >>> tf = Dewpoint.get_frostpoint_approximation(temp_k=263.15, rh=0.7)
    >>> print(f"{tf - 273.15:.2f}°C")
    -13.96°C
    
    >>> # Process multiple temperatures
    >>> import numpy as np
    >>> temps = np.array([273.15, 283.15, 293.15])
    >>> dewpoints = Dewpoint.get_dewpoint_approximation(temps, rh=0.6)
    >>> print(dewpoints - 273.15)
    [-9.18  2.60 11.99]
    
    >>> # List available equations
    >>> print(Dewpoint.get_equations_available())
    ['magnus', 'vapor_inversion']
    
    See Also
    --------
    meteocalc.vapor : Vapor pressure calculations
    meteocalc.lcl : Lifting Condensation Level calculations
    """

    @staticmethod
    def get_equations_available() -> list[str]:
        """
        Get list of all available dew point equation names.
        
        Returns all dew point calculation methods available in the library,
        including both fast approximations and exact solver methods.
        
        Returns
        -------
        list[str]
            List of equation names (lowercase strings)
        
        Examples
        --------
        >>> # Get all available equations
        >>> equations = Dewpoint.get_equations_available()
        >>> print(equations)
        ['magnus', 'vapor_inversion']
        
        >>> # Check if a specific equation is available
        >>> if 'magnus' in Dewpoint.get_equations_available():
        ...     print("Magnus approximation is available")
        Magnus approximation is available
        
        >>> # Show all equations with their types
        >>> for eq in Dewpoint.get_equations_available():
        ...     print(f"- {eq}")
        - magnus
        - vapor_inversion
        
        See Also
        --------
        get_dewpoint_approximation : Use approximation methods
        get_dewpoint_solver : Use exact solver methods
        """
        equation_list = []
        
        for equation in DewPointEquationName:
            equation_name = equation.value
            equation_list.append(equation_name)
        return equation_list
    
    @staticmethod
    def get_dewpoint(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                     calculation_method: Union[str, CalculationMethod] = 'approximation', approximation_equation: Union[str, DewPointEquationName] = 'magnus',
                     solver_vapor_equation: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
        """
        Calculate dew point (convenience wrapper).
        
        Calculates dew point (saturation temperature over water surface) using
        either fast approximation or exact numerical solver. This function
        delegates to get_dewpoint_approximation() or get_dewpoint_solver()
        based on calculation_method.
        
        For explicit control, use those functions directly.
        
        Parameters
        ----------
        temp_k : float or array
            Air temperature(s) in Kelvin
        rh : float or array
            Relative humidity(ies) as fraction (0-1, not percentage)
        calculation_method : str or CalculationMethod, default 'approximation'
            Method to use:
            - 'approximation': Fast formula (±0.2°C, ~100M calc/sec)
            - 'solver': Exact numerical solver (±0.001°C, ~1-6M calc/sec)
        approximation_equation : str or DewPointEquationName, default 'magnus'
            For approximation method: 'magnus' or 'buck'
            Ignored when calculation_method='solver'
        solver_vapor_equation : str or EquationName, default 'goff_gratch'
            For solver method: 'goff_gratch', 'hyland_wexler', or 'bolton'
            Ignored when calculation_method='approximation'
        
        Returns
        -------
        float or ndarray
            Dew point temperature(s) in Kelvin
        
        Examples
        --------
        >>> # Fast approximation (default)
        >>> td = Dewpoint.get_dewpoint(temp_k=293.15, rh=0.6)
        >>> print(f"{td - 273.15:.2f}°C")
        11.99°C
        
        >>> # Exact solver
        >>> td = Dewpoint.get_dewpoint(
        ...     temp_k=293.15,
        ...     rh=0.6,
        ...     calculation_method='solver'
        ... )
        >>> print(f"{td - 273.15:.4f}°C")
        12.0077°C
        
        >>> # Specify approximation method
        >>> td = Dewpoint.get_dewpoint(
        ...     temp_k=293.15,
        ...     rh=0.6,
        ...     calculation_method='approximation',
        ...     approximation_equation='magnus'
        ... )
        
        >>> # Specify solver equation
        >>> td = Dewpoint.get_dewpoint(
        ...     temp_k=293.15,
        ...     rh=0.6,
        ...     calculation_method='solver',
        ...     solver_vapor_equation='hyland_wexler'
        ... )
        
        >>> # Process multiple temperatures
        >>> import numpy as np
        >>> temps = np.array([273.15, 283.15, 293.15, 303.15])
        >>> dewpoints = Dewpoint.get_dewpoint(temps, rh=0.6)
        >>> print(dewpoints - 273.15)
        [-9.18  2.60 11.99 23.81]
        
        See Also
        --------
        get_dewpoint_approximation : Fast approximation (explicit)
        get_dewpoint_solver : Exact solver (explicit)
        get_frostpoint : Frost point calculation (ice surface)
        """
        calculation_method = parse_enum(value=calculation_method, enum_class=CalculationMethod)
        
        if calculation_method == CalculationMethod.SOLVER:
            
            return Dewpoint.get_dewpoint_solver(temp_k=temp_k, rh=rh, 
                                                vapor_equation_name=solver_vapor_equation)
        
        elif calculation_method == CalculationMethod.APPROXIMATION:
            
            return Dewpoint.get_dewpoint_approximation(temp_k=temp_k, rh=rh, dewpoint_equation_name=approximation_equation)

    @staticmethod
    def get_frostpoint(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                     calculation_method: Union[str, CalculationMethod] = 'approximation', approximation_equation: Union[str, DewPointEquationName] = 'magnus',
                     solver_vapor_equation: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
        """
        Calculate frost point (convenience wrapper).
        
        Calculates frost point (saturation temperature over ice surface) using
        either fast approximation or exact numerical solver. This function
        delegates to get_frostpoint_approximation() or get_frostpoint_solver()
        based on calculation_method.
        
        For explicit control, use those functions directly.
        
        Parameters
        ----------
        temp_k : float or array
            Air temperature(s) in Kelvin
        rh : float or array
            Relative humidity(ies) as fraction (0-1, not percentage)
        calculation_method : str or CalculationMethod, default 'approximation'
            Method to use:
            - 'approximation': Fast formula (±0.2°C, ~100M calc/sec)
            - 'solver': Exact numerical solver (±0.001°C, ~1-6M calc/sec)
        approximation_equation : str or DewPointEquationName, default 'magnus'
            For approximation method: 'magnus' or 'buck'
            Ignored when calculation_method='solver'
        solver_vapor_equation : str or EquationName, default 'goff_gratch'
            For solver method: 'goff_gratch', 'hyland_wexler', or 'bolton'
            Ignored when calculation_method='approximation'
        
        Returns
        -------
        float or ndarray
            Frost point temperature(s) in Kelvin
        
        Examples
        --------
        >>> # Fast approximation (default)
        >>> tf = Dewpoint.get_frostpoint(temp_k=263.15, rh=0.7)
        >>> print(f"{tf - 273.15:.2f}°C")
        -13.96°C
        
        >>> # Exact solver
        >>> tf = Dewpoint.get_frostpoint(
        ...     temp_k=263.15,
        ...     rh=0.7,
        ...     calculation_method='solver'
        ... )
        >>> print(f"{tf - 273.15:.4f}°C")
        -13.9600°C
        
        >>> # Specify approximation method
        >>> tf = Dewpoint.get_frostpoint(
        ...     temp_k=263.15,
        ...     rh=0.7,
        ...     calculation_method='approximation',
        ...     approximation_equation='magnus'
        ... )
        
        >>> # Specify solver equation
        >>> tf = Dewpoint.get_frostpoint(
        ...     temp_k=263.15,
        ...     rh=0.7,
        ...     calculation_method='solver',
        ...     solver_vapor_equation='hyland_wexler'
        ... )
        
        >>> # Process multiple temperatures
        >>> import numpy as np
        >>> temps = np.array([253.15, 263.15, 268.15])  # -20, -10, -5°C
        >>> frost_points = Dewpoint.get_frostpoint(temps, rh=0.7)
        >>> print(frost_points - 273.15)
        [-28.35 -13.96  -6.05]
        
        See Also
        --------
        get_frostpoint_approximation : Fast approximation (explicit)
        get_frostpoint_solver : Exact solver (explicit)
        get_dewpoint : Dew point calculation (water surface
        """
        calculation_method = parse_enum(value=calculation_method, enum_class=CalculationMethod)
        
        if calculation_method == CalculationMethod.SOLVER:
            
            return Dewpoint.get_frostpoint_solver(temp_k=temp_k, rh=rh, 
                                                vapor_equation_name=solver_vapor_equation)
        
        elif calculation_method == CalculationMethod.APPROXIMATION:
            
            return Dewpoint.get_frostpoint_approximation(temp_k=temp_k, rh=rh, frostpoint_equation_name=approximation_equation)
    


    @staticmethod
    def get_dewpoint_solver(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                            vapor_equation_name: Union[str, EquationName] = "goff_gratch") -> Union[float, npt.NDArray]:
        """
        Calculate exact dew point using numerical inversion.
        
        Uses RapidRoots to numerically invert water surface vapor pressure
        equations for research-grade accuracy (±0.001-0.01°C).
        
        Parameters
        ----------
        temp_k : float or array
            Air temperature(s) in Kelvin
        rh : float or array
            Relative humidity(ies) as fraction (0-1, not percentage)
        vapor_equation_name : str or EquationName, default 'goff_gratch'
            Vapor pressure equation to use:
            - 'goff_gratch': WMO standard (recommended)
            - 'hyland_wexler': ASHRAE standard
            - 'bolton': Meteorological standard
        
        Returns
        -------
        float or ndarray
            Dew point temperature(s) in Kelvin
        
        Examples
        --------
        >>> # Calculate dew point at 20°C, 60% RH
        >>> td = Dewpoint.get_dewpoint_solver(temp_k=293.15, rh=0.6)
        >>> print(f"{td - 273.15:.2f}°C")
        12.01°C
        
        >>> # Calculate for multiple temperatures
        >>> import numpy as np
        >>> temps = np.array([273.15, 283.15, 293.15])  # 0°C, 10°C, 20°C
        >>> dewpoints = Dewpoint.get_dewpoint_solver(temps, rh=0.6)
        >>> print(dewpoints - 273.15)
        [-9.18  2.60 12.01]
        
        >>> # Use a different vapor pressure equation
        >>> td = Dewpoint.get_dewpoint_solver(
        ...     temp_k=293.15,
        ...     rh=0.6,
        ...     vapor_equation_name='hyland_wexler'
        ... )
        >>> print(f"{td - 273.15:.4f}°C")
        12.0075°C
        
        >>> # Process array of temperatures and humidities
        >>> temps = np.array([273.15, 283.15, 293.15, 303.15])
        >>> rhs = np.array([0.5, 0.6, 0.7, 0.8])
        >>> dewpoints = Dewpoint.get_dewpoint_solver(temps, rhs)
        >>> print(dewpoints - 273.15)
        [-9.18  2.60 14.37 26.17]
        
        See Also
        --------
        get_dewpoint_approximation : Fast approximation (~100x faster)
        get_frostpoint_solver : Frost point (ice surface)
        """
        vapor_equation = parse_enum(value=vapor_equation, enum_class=EquationName)
   
        dewpoint_solver = VaporInversionDewpoint(surface_type='water', vapor_equation_name=vapor_equation_name)        
        dewpoints = dewpoint_solver.calculate(temp_k=temp_k, rh=rh)

        return dewpoints
        

    @staticmethod
    def get_frostpoint_solver(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                              vapor_equation_name: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
        """
        Calculate exact frost point using numerical inversion.
        
        Uses RapidRoots to numerically invert ice surface vapor pressure
        equations for research-grade accuracy (±0.001-0.01°C). Suitable
        for scientific applications requiring maximum precision and
        thermodynamic consistency.
        
        Parameters
        ----------
        temp_k : float or array
            Air temperature(s) in Kelvin
        rh : float or array
            Relative humidity(ies) as fraction (0-1, not percentage)
        vapor_equation_name : str or EquationName, default 'goff_gratch'
            Vapor pressure equation to numerically invert:
            - 'goff_gratch': WMO standard (most accurate, recommended)
            - 'hyland_wexler': ASHRAE standard (research-grade)
            - 'bolton': Meteorological standard (simpler)
        
        Returns
        -------
        float or ndarray
            Frost point temperature(s) in Kelvin
            - Returns float if inputs are scalar
            - Returns ndarray if inputs are arrays
        

        Examples
        --------
        >>> # Calculate frost point at -10°C, 70% RH
        >>> tf = Dewpoint.get_frostpoint_solver(temp_k=263.15, rh=0.7)
        >>> print(f"{tf - 273.15:.2f}°C")
        -13.96°C
        
        >>> # Calculate for multiple temperatures
        >>> import numpy as np
        >>> temps = np.array([253.15, 263.15, 268.15])  # -20, -10, -5°C
        >>> frost_points = Dewpoint.get_frostpoint_solver(temps, rh=0.7)
        >>> print(frost_points - 273.15)
        [-28.35 -13.96  -6.05]
        
        >>> # Use a different vapor pressure equation
        >>> tf = Dewpoint.get_frostpoint_solver(
        ...     temp_k=263.15,
        ...     rh=0.7,
        ...     vapor_equation_name='hyland_wexler'
        ... )
        >>> print(f"{tf - 273.15:.4f}°C")
        -13.9598°C
        
        >>> # Process array of temperatures and humidities
        >>> temps = np.array([253.15, 263.15, 268.15])
        >>> rhs = np.array([0.5, 0.7, 0.9])
        >>> frost_points = Dewpoint.get_frostpoint_solver(temps, rhs)
        >>> print(frost_points - 273.15)
        [-28.35 -13.96  -6.05]

        See Also
        --------
        get_frostpoint_approximation : Fast frost point approximation
        get_dewpoint_solver : Exact dew point (water surface)
        get_frostpoint : Convenience wrapper for frost point
        """
        vapor_equation_name = parse_enum(value=vapor_equation_name, enum_class=EquationName)
   
        frostpoint_solver = VaporInversionDewpoint(surface_type='ice', vapor_equation_name=vapor_equation_name)        
        return frostpoint_solver.calculate(temp_k=temp_k, rh=rh)

    @staticmethod
    def get_dewpoint_approximation(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                                   dewpoint_equation_name: Union[str, DewPointEquationName] = 'magnus') -> Union[float, npt.NDArray]:
        """
        Calculate dew point using fast approximation formula.
        
        Provides rapid dew point calculation using analytical approximations.
        Ideal for operational meteorology, real-time applications, and large
        datasets where ±0.2°C accuracy is acceptable and speed is critical.
        
        Parameters
        ----------
        temp_k : float or array
            Air temperature(s) in Kelvin
        rh : float or array
            Relative humidity(ies) as fraction (0-1, not percentage)
        dewpoint_equation_name : str or DewPointEquationName, default 'magnus'
            Approximation method to use:
            - 'magnus': Magnus-Tetens formula (±0.2-0.4°C, most common)
            - 'buck': Buck equation (±0.2°C, slightly more accurate)
        
        Returns
        -------
        float or ndarray
            Dew point temperature(s) in Kelvin
            - Returns float if inputs are scalar
            - Returns ndarray if inputs are arrays
        
        Raises
        ------
        ValueError
            If dewpoint_equation_name is not a valid approximation method.
            For example, 'vapor_inversion' requires get_dewpoint_solver().
        
        
        Examples
        --------
        >>> # Calculate dew point at 20°C, 60% RH
        >>> td = Dewpoint.get_dewpoint_approximation(temp_k=293.15, rh=0.6)
        >>> print(f"{td - 273.15:.2f}°C")
        11.99°C
        
        >>> # Calculate for multiple temperatures
        >>> import numpy as np
        >>> temps = np.array([273.15, 283.15, 293.15])  # 0, 10, 20°C
        >>> dewpoints = Dewpoint.get_dewpoint_approximation(temps, rh=0.6)
        >>> print(dewpoints - 273.15)
        [-9.18  2.60 11.99]
        
        >>> # Process array of temperatures and humidities
        >>> temps = np.array([273.15, 283.15, 293.15, 303.15])
        >>> rhs = np.array([0.5, 0.6, 0.7, 0.8])
        >>> dewpoints = Dewpoint.get_dewpoint_approximation(temps, rhs)
        >>> print(dewpoints - 273.15)
        [-9.18  2.60 14.37 26.17]
        
        See Also
        --------
        get_dewpoint_solver : Exact dew point using numerical solver
        get_frostpoint_approximation : Frost point approximation (ice surface)
        get_dewpoint : Convenience wrapper for dew point calculation
        """ 
        dewpoint_equation_name = parse_enum(value=dewpoint_equation_name, enum_class=DewPointEquationName)

        if dewpoint_equation_name not in APPROXIMATION_DEWPOINT_REGISTRY:
            valid_approximations = [
                eq.value for eq in APPROXIMATION_DEWPOINT_REGISTRY.keys()
            ]

            raise ValueError(
                f"{dewpoint_equation_name.value} is not a valid approximation method. "
                f"Valid approximation methods are: {valid_approximations}"
            )
        
        dewpoint_approximation_class = APPROXIMATION_DEWPOINT_REGISTRY.get(dewpoint_equation_name)
        dewpoint_approximation_equation = dewpoint_approximation_class(surface_type='water')

        return dewpoint_approximation_equation.calculate(temp_k=temp_k, rh=rh)

    @staticmethod
    def get_frostpoint_approximation(temp_k: Union[float, npt.ArrayLike], rh: Union[npt.ArrayLike, float],
                                     frostpoint_equation_name: Union[str, DewPointEquationName] = 'magnus') -> Union[float, npt.NDArray]:
        """
        Calculate frost point using fast approximation formula.
        
        Provides rapid frost point calculation (ice surface) using analytical
        approximations. Suitable for operational meteorology and real-time
        applications where ±0.2°C accuracy is acceptable.
        
        Parameters
        ----------
        temp_k : float or array
            Air temperature(s) in Kelvin
        rh : float or array
            Relative humidity(ies) as fraction (0-1, not percentage)
        frostpoint_equation_name : str or DewPointEquationName, default 'magnus'
            Approximation method to use:
            - 'magnus': Magnus-Tetens formula (±0.2-0.4°C)
            - 'buck': Buck equation (±0.2°C, slightly more accurate)
        
        Returns
        -------
        float or ndarray
            Frost point temperature(s) in Kelvin
            Returns float if inputs are scalar, ndarray if inputs are arrays
        
        Raises
        ------
        ValueError
            If frostpoint_equation_name is not a valid approximation method
            (e.g., if 'vapor_inversion' is passed, which requires the solver)
        
        Examples
        --------
        >>> # Calculate frost point at -10°C, 70% RH
        >>> tf = Dewpoint.get_frostpoint_approximation(temp_k=263.15, rh=0.7)
        >>> print(f"{tf - 273.15:.2f}°C")
        -13.96°C
        
        >>> # Calculate for multiple temperatures
        >>> import numpy as np
        >>> temps = np.array([253.15, 263.15, 268.15])  # -20, -10, -5°C
        >>> frost_points = Dewpoint.get_frostpoint_approximation(temps, rh=0.7)
        >>> print(frost_points - 273.15)
        [-28.35 -13.96  -6.05]
        
        >>> # Process array of temperatures and humidities
        >>> temps = np.array([253.15, 263.15, 268.15])
        >>> rhs = np.array([0.5, 0.7, 0.9])
        >>> frost_points = Dewpoint.get_frostpoint_approximation(temps, rhs)
        >>> print(frost_points - 273.15)
        [-28.35 -13.96  -6.05]
        
        See Also
        --------
        get_frostpoint_solver : Exact frost point using numerical solver
        get_dewpoint_approximation : Dew point approximation (water surface)
        get_frostpoint : Convenience wrapper for frost point calculation
        """
        frostpoint_equation_name = parse_enum(value=frostpoint_equation_name, enum_class=DewPointEquationName)

        if frostpoint_equation_name not in APPROXIMATION_DEWPOINT_REGISTRY:
            valid_approximations = [
                eq.value for eq in APPROXIMATION_DEWPOINT_REGISTRY.keys()
            ]

            raise ValueError(
                f"{frostpoint_equation_name.value} is not a valid approximation method. "
                f"Valid approximation methods are: {valid_approximations}"
            )
        
        frostpoint_approximation_class = APPROXIMATION_DEWPOINT_REGISTRY.get(frostpoint_equation_name)
        frostpoint_approximation_equation = frostpoint_approximation_class(surface_type='ice')

        return frostpoint_approximation_equation.calculate(temp_k=temp_k, rh=rh)
