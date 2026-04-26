"""
Calculates the lcl using solver method.

Author: Cian Quezon
"""

from typing import Union

import numpy as np
import numpy.typing as npt
from numba import njit
from rapid_roots.solvers import RootSolvers

from meteocalc.shared.constants import Rd, cpd, eps
from meteocalc.vapor._vapor_equations import (
    VaporEquation,
)


def get_lcl_using_solver(
    temp_k: Union[float, npt.ArrayLike],
    dewpoint_k: Union[float, npt.ArrayLike],
    pressure_hpa: Union[float, npt.ArrayLike],
    mixing_ratio: Union[float, npt.ArrayLike],
    vapor_equation: VaporEquation,
):
    """
    Compute the Lifting Condensation Level (LCL) temperature using an
    iterative root-finding solver.

    Finds the temperature at which an air parcel becomes saturated when
    lifted dry-adiabatically from the surface by solving:

        f(T_LCL) = rs(T_LCL, p_LCL) - w = 0

    where rs is the saturation mixing ratio at the LCL and w is the
    surface mixing ratio. The LCL pressure is derived from the dry
    adiabatic Poisson relation:

        p_LCL = p_surface * (T_LCL / T_surface) ^ (cpd / Rd)

    The bracket for the root search is physically motivated:
        a = Td - (T - Td)   lower bound — below T_LCL by construction
        b = T               upper bound — parcel is unsaturated at surface

    Scalar inputs return scalar outputs. Array inputs return arrays.
    Mixed scalar/array inputs are not supported — all inputs must be
    the same shape or all scalar.

    Parameters
    ----------
    temp_k : float or array_like of float
        Surface air temperature in Kelvin. Must be greater than
        ``dewpoint_k`` for all parcels.
    dewpoint_k : float or array_like of float
        Surface dewpoint temperature in Kelvin. Must be less than
        ``temp_k`` for all parcels.
    pressure_hpa : float or array_like of float
        Surface pressure in hectopascals (hPa). Typical range
        850–1050 hPa.
    mixing_ratio : float or array_like of float
        Surface water vapour mixing ratio in kg/kg. Should be
        computed at ``dewpoint_k`` using the same ``vapor_equation``
        for physical consistency:

            e  = vapor_equation.calculate(dewpoint_k)
            w  = eps * e / (pressure_hpa - e)

    vapor_equation : VaporEquation
        Saturation vapour pressure formulation used in the objective
        function. Must be a concrete subclass of ``VaporEquation``
        (e.g. ``GoffGratchEquation``, ``HylandWexlerEquation``).
        The same equation should be used to compute ``mixing_ratio``
        to ensure physical consistency.

    Returns
    -------
    lcl_temp_k : float or ndarray of float64
        LCL temperature(s) in Kelvin. Returns a Python ``float``
        if all inputs were scalar, otherwise ``np.ndarray`` of
        shape ``(n,)``.
    iterations : int or ndarray of int32
        Number of solver iterations used per parcel. Returns a
        Python ``int`` if all inputs were scalar, otherwise
        ``np.ndarray`` of shape ``(n,)``.
    converged : bool or ndarray of bool
        Convergence flag per parcel. ``True`` if the solver found
        a root within tolerance, ``False`` otherwise. Returns a
        Python ``bool`` if all inputs were scalar, otherwise
        ``np.ndarray`` of shape ``(n,)``.

    Warns
    -----
    UserWarning
        If any parcels fail to converge. Check that ``temp_k > dewpoint_k``
        for all inputs and that ``vapor_equation`` temperature bounds
        are satisfied.


    Examples
    --------
    Scalar input — standard mid-latitude case:

    >>> from meteocalc.vapor._vapor_equations import GoffGratchEquation
    >>> from meteocalc.shared.constants import eps
    >>> vapor_eq = GoffGratchEquation(surface_type="water")
    >>> temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
    >>> e = float(vapor_eq.calculate(dewpoint_k))
    >>> w = eps * e / (pressure_hpa - e)
    >>> lcl_temp, iters, converged = get_lcl_using_solver(
    ...     temp_k=temp_k,
    ...     dewpoint_k=dewpoint_k,
    ...     pressure_hpa=pressure_hpa,
    ...     mixing_ratio=w,
    ...     vapor_equation=vapor_eq,
    ... )
    >>> print(f"T_LCL = {lcl_temp:.4f} K ({lcl_temp - 273.15:.2f} °C)")
    T_LCL = 283.3605 K (10.21 °C)
    >>> print(f"Converged in {iters} iterations: {converged}")
    Converged in 6 iterations: True

    Array input — multiple parcels:

    >>> import numpy as np
    >>> temp_arr     = np.array([293.15, 303.15, 283.15])
    >>> dewpoint_arr = np.array([285.15, 298.15, 278.15])
    >>> pressure_arr = np.full(3, 1013.25)
    >>> e_arr = vapor_eq.calculate(dewpoint_arr).astype(np.float64)
    >>> w_arr = eps * e_arr / (pressure_arr - e_arr)
    >>> lcl_temps, iters_arr, converged_arr = get_lcl_using_solver(
    ...     temp_k=temp_arr,
    ...     dewpoint_k=dewpoint_arr,
    ...     pressure_hpa=pressure_arr,
    ...     mixing_ratio=w_arr,
    ...     vapor_equation=vapor_eq,
    ... )
    >>> print(lcl_temps)
    [283.3605 296.9407 277.0650]


    """

    was_scalar = np.ndim(temp_k) == 0 and np.ndim(dewpoint_k) == 0

    temp_k = np.atleast_1d(np.asarray(temp_k, dtype=np.float64))
    dewpoint_k = np.atleast_1d(np.asarray(dewpoint_k, dtype=np.float64))
    pressure_hpa = np.atleast_1d(np.asarray(pressure_hpa, dtype=np.float64))
    mixing_ratio = np.atleast_1d(np.asarray(mixing_ratio, dtype=np.float64))

    a = dewpoint_k - (temp_k - dewpoint_k)
    b = temp_k

    lcl_objective_func = _get_lcl_objective_function(vapor_equation=vapor_equation)

    func_params = np.column_stack([mixing_ratio, pressure_hpa, temp_k])

    roots, iters, converged = RootSolvers.get_root(
        func=lcl_objective_func,
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


def _get_lcl_objective_function(vapor_equation: VaporEquation):
    """
    Build a JIT-compiled LCL objective function for a given vapour equation.

    Captures the JIT scalar function and constants from ``vapor_equation``
    in a Numba closure. The returned function evaluates:

        f(x) = rs(x, p_LCL) - w

    where rs is the saturation mixing ratio at the candidate LCL
    temperature x and p_LCL is derived from the Poisson relation.
    The root of f is the LCL temperature.

    Parameters
    ----------
    vapor_equation : VaporEquation
        Saturation vapour pressure formulation to use.

    Returns
    -------
    lcl_objective : callable
        Numba ``@njit`` scalar function with signature:
        ``(x, mixing_ratio, surface_pressure, surface_temp) -> float``
    """

    vapor_scalar_func = vapor_equation.get_jit_scalar_func()
    surface_constants = vapor_equation.get_constants()
    tuple_surface_constants = tuple(surface_constants)

    @njit
    def lcl_objective(
        x: float, mixing_ratio: float, surface_pressure: float, surface_temp: float
    ):
        """
        Scalar LCL objective function for root-finding.

        Parameters
        ----------
        x : float
            Candidate LCL temperature in Kelvin. This is the variable
            being solved for — the root of this function is T_LCL.
        mixing_ratio : float
            Surface water vapour mixing ratio in kg/kg.
        surface_pressure : float
            Surface pressure in hPa.
        surface_temp : float
            Surface air temperature in Kelvin. Used in the Poisson
            relation to compute p_LCL.

        Returns
        -------
        float
            Residual rs(x) - w in kg/kg. Zero when x = T_LCL.
        """

        p_lcl = surface_pressure * (x / surface_temp) ** (cpd / Rd)
        es = vapor_scalar_func(x, *tuple_surface_constants)
        rs = eps * es / (p_lcl - es)

        return rs - mixing_ratio

    return lcl_objective
