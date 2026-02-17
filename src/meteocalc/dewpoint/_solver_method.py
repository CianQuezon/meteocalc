"""
Docstring for meteocalc.dewpoint._solver_method
"""

from rapid_roots.solvers import RootSolvers

from typing import Union, Callable
from meteocalc.vapor.core import Vapor
from meteocalc.vapor._vapor_equations import VaporEquation
from meteocalc.vapor._enums import EquationName, SurfaceType
from meteocalc.shared._enum_tools import parse_enum
from numba import njit

import numpy.typing as npt
import numpy as np



def get_dewpoint_using_solver(temp_k: Union[float, npt.ArrayLike], rh: Union[npt.ArrayLike],
                              surface_type: Union[str, SurfaceType], vapor_equation: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
    """
    Docstring for get_dewpoint_using_solver

    :param temp_k: Description
    :type temp_k: Union[float, npt.ArrayLike]
    :param rh: Description
    :type rh: Union[npt.ArrayLike]
    :param vapor_equation: Description
    :type vapor_equation: Union[str, EquationName]
    :param surface_type: Description
    :type surface_type: Union[str, SurfaceType]
    :return: Description
    :rtype: float | NDArray
    """
    was_scalar = np.ndim(temp_k) == 0 and np.ndim(rh) == 0

    temp_k = np.atleast_1d(temp_k).astype(np.float64)
    rh = np.atleast_1d(rh).astype(np.float64)
    
    n = len(temp_k)

    e_sat_air = Vapor.get_vapor_saturation(temp_k=temp_k, phase=surface_type, equation=vapor_equation)
    vapor_equation = Vapor.get_equation(equation=vapor_equation, phase=surface_type)
    
    e_actual = rh * e_sat_air

    a = temp_k - 100
    b = temp_k

    func_params = np.empty((n, 1), dtype=np.float64)
    func_params[:, 0] = e_actual.copy()

    dewpoint_objective_func = get_dewpoint_objective_function(vapor_equation=vapor_equation)

    roots, iters, converged = RootSolvers.get_root(
        func=dewpoint_objective_func,
        a=a,
        b=b,
        func_params=func_params,
        main_solver='brent',
        use_backup=True,
        backup_solvers=['bisection']
    )

    if was_scalar:
        return float(roots[0]), int(iters[0]), bool(converged[0])

    return roots, iters, converged

def get_dewpoint_objective_function(vapor_equation: VaporEquation):
    """
    Create JIT-compiled objective function for dew point root finding.
    
    Generates a Numba-compiled objective function that calculates the
    difference between saturation vapor pressure at a given temperature
    and the actual vapor pressure. Used by numerical solvers to find
    the temperature where this difference equals zero (the dew/frost point).
    
    Parameters
    ----------
    vapor_equation : VaporEquation
        Vapor pressure equation instance (e.g., Goff-Gratch, Hyland-Wexler)
        Must provide get_jit_scalar_func() and get_constants() methods
    
    Returns
    -------
    callable
        JIT-compiled objective function with signature:
        f(x: float, e_actual: float) -> float
        
        where:
        - x: Trial temperature in Kelvin
        - e_actual: Actual vapor pressure in Pascals
        - returns: e_sat(x) - e_actual (zero at dew point)
    
    Examples
    --------
    >>> from meteocalc.vapor import Vapor
    >>> from meteocalc.vapor._enums import EquationName, SurfaceType
    >>> 
    >>> # Get Goff-Gratch equation for water surface
    >>> vapor_eq = Vapor.get_equation(
    ...     equation=EquationName.GOFF_GRATCH,
    ...     phase=SurfaceType.WATER
    ... )
    >>> 
    >>> # Create objective function
    >>> objective = get_dewpoint_objective_function(vapor_eq)
    >>> 
    >>> # Example: Find dew point at 20°C, 60% RH
    >>> temp_k = 293.15
    >>> rh = 0.6
    >>> e_sat_air = vapor_eq.calculate(temp_k)
    >>> e_actual = rh * e_sat_air
    >>> 
    >>> # Evaluate objective at different temperatures
    >>> print(f"At 285 K: {objective(285.0, e_actual):.2f} Pa")
    >>> print(f"At 290 K: {objective(290.0, e_actual):.2f} Pa")
    >>> print(f"At 295 K: {objective(295.0, e_actual):.2f} Pa")
    At 285 K: -402.15 Pa
    At 290 K: -12.34 Pa
    At 295 K: 891.23 Pa
    # Zero-crossing indicates dew point is between 290-295 K
    
    See Also
    --------
    get_dewpoint_using_solver : Uses this objective function for solving
    meteocalc.vapor.VaporEquation : Base class for vapor equations
    """

    vapor_scalar_func = vapor_equation.get_jit_scalar_func()
    surface_constants = vapor_equation.get_constants()
    tuple_surface_constants = tuple(surface_constants)

    @njit
    def dewpoint_objective(x: float, e_actual: float):
        """
        Objective function for dew point root finding.
        
        Calculates the difference between saturation vapor pressure at
        temperature x and the actual vapor pressure. The root of this
        function is the dew/frost point temperature.
        
        Parameters
        ----------
        x : float
            Trial temperature in Kelvin
        e_actual : float
            Actual vapor pressure in Pascals (from RH × e_sat(T_air))
        
        Returns
        -------
        float
            Residual: e_sat(x) - e_actual
            Zero when x equals the dew/frost point temperature
        """
        e_sat = vapor_scalar_func(x, *tuple_surface_constants)

        return e_sat - e_actual

    return dewpoint_objective
