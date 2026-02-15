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

    vapor_scalar_func = vapor_equation.get_jit_scalar_func()
    surface_constants = vapor_equation.get_constants()
    tuple_surface_constants = tuple(surface_constants)

    @njit
    def dewpoint_objective(x: float, e_actual: float):
        """
        Docstring for dewpoint_objective
        
        :param x: Description
        :param e_sat: Description
        """
        e_sat = vapor_scalar_func(x, *tuple_surface_constants)

        return e_sat - e_actual

    return dewpoint_objective
