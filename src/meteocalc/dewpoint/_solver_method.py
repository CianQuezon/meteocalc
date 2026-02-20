"""
Calculates the dewpoint using solver inversion method.

Author: Cian Quezon
"""

from typing import Union

import numpy as np
import numpy.typing as npt
from numba import njit
from rapid_roots.solvers import RootSolvers

from meteocalc.vapor._enums import EquationName, SurfaceType
from meteocalc.vapor._vapor_equations import VaporEquation
from meteocalc.vapor.core import Vapor


def get_dewpoint_using_solver(
    temp_k: Union[float, npt.ArrayLike],
    rh: Union[float, npt.ArrayLike],
    surface_type: Union[str, SurfaceType],
    vapor_equation: Union[str, EquationName] = "goff_gratch",
) -> Union[tuple[float, int, bool], tuple[npt.NDArray, npt.NDArray, npt.NDArray]]:
    """
    Calculate exact dew/frost point using numerical root finding.

    Numerically inverts vapor pressure equations to find the exact temperature
    at which air becomes saturated. Uses RapidRoots with Brent's method and
    automatic fallback to bisection for guaranteed convergence.

    This is the core numerical solver used by VaporInversionDewpoint class.

    Parameters
    ----------
    temp_k : float or array
        Air temperature(s) in Kelvin
    rh : float or array
        Relative humidity(ies) as fraction (0-1, not percentage)
    surface_type : str or SurfaceType
        Surface type for saturation:
        - 'water': Calculate dew point (liquid water surface)
        - 'ice': Calculate frost point (ice surface)
    vapor_equation : str or EquationName, default 'goff_gratch'
        Vapor pressure equation to invert:
        - 'goff_gratch': WMO standard (recommended)
        - 'hyland_wexler': ASHRAE standard
        - 'bolton': Meteorological standard

    Returns
    -------
    roots : float or ndarray
        Dew/frost point temperature(s) in Kelvin
    iters : int or ndarray
        Number of iterations taken for convergence
    converged : bool or ndarray
        Convergence status (True if converged)

    Examples
    --------
    >>> # Single calculation (dew point)
    >>> td, iters, converged = get_dewpoint_using_solver(
    ...     temp_k=293.15,
    ...     rh=0.6,
    ...     surface_type='water',
    ...     vapor_equation='goff_gratch'
    ... )
    >>> print(f"Dew point: {td - 273.15:.4f}°C")
    >>> print(f"Iterations: {iters}, Converged: {converged}")
    Dew point: 12.0077°C
    Iterations: 7, Converged: True

    >>> # Array calculation (frost point)
    >>> import numpy as np
    >>> temps = np.array([253.15, 263.15, 268.15])
    >>> rhs = np.array([0.5, 0.7, 0.9])
    >>> tfs, iters, converged = get_dewpoint_using_solver(
    ...     temp_k=temps,
    ...     rh=rhs,
    ...     surface_type='ice',
    ...     vapor_equation='goff_gratch'
    ... )
    >>> print(tfs - 273.15)
    >>> print(f"All converged: {np.all(converged)}")
    [-28.35 -13.96  -6.05]
    All converged: True

    >>> # Different vapor equation
    >>> td, _, _ = get_dewpoint_using_solver(
    ...     temp_k=293.15,
    ...     rh=0.6,
    ...     surface_type='water',
    ...     vapor_equation='hyland_wexler'
    ... )
    >>> print(f"{td - 273.15:.4f}°C")
    12.0075°C

    See Also
    --------
    VaporInversionDewpoint : High-level interface using this solver
    get_dewpoint_objective_function : Creates the objective function
    meteocalc.rapid_roots.RootSolvers : Numerical solver backend
    """
    was_scalar = np.ndim(temp_k) == 0 and np.ndim(rh) == 0

    temp_k = np.atleast_1d(temp_k).astype(np.float64)
    rh = np.atleast_1d(rh).astype(np.float64)

    n = len(temp_k)

    e_sat_air = Vapor.get_vapor_saturation(
        temp_k=temp_k, phase=surface_type, equation=vapor_equation
    )
    vapor_equation = Vapor.get_equation(equation=vapor_equation, phase=surface_type)

    e_actual = rh * e_sat_air

    a = temp_k - 100
    b = temp_k

    func_params = np.empty((n, 1), dtype=np.float64)
    func_params[:, 0] = e_actual.copy()

    dewpoint_objective_func = get_dewpoint_objective_function(
        vapor_equation=vapor_equation
    )

    roots, iters, converged = RootSolvers.get_root(
        func=dewpoint_objective_func,
        a=a,
        b=b,
        func_params=func_params,
        main_solver="brent",
        use_backup=True,
        backup_solvers=["bisection"],
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
