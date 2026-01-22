"""
Codegen function for vectorised solvers

Author: Cian Quezon
"""

from typing import Callable, Union

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from meteorological_equations.math.solvers._enums import MethodType
from meteorological_equations.shared._enum_tools import parse_enum


def generate_vectorised_solver(
    scalar_func: Callable[[float], float], num_params: int, method_type: Union[MethodType, str]
) -> Callable[[npt.ArrayLike], npt.NDArray[np.float64]]:
    """
    Generate a JIT-compiled vectorised solver from a scalar solver function.
    
    This function creates specialized vectorised code at runtime to solve the
    problem of dynamic parameter unpacking in Numba's prange loops. Since
    Numba's parallel loops cannot dynamically unpack variable-length tuples
    (e.g., `*func_params`), this code generator creates a version with the
    exact number of parameters needed, enabling both parallelization and
    flexibility.
    
    Parameters
    ----------
    scalar_func : callable
        Scalar solver function to be vectorised. Must be JIT-compiled with
        @njit and have one of these signatures:
        
        - Open methods: func(f, f_prime, x0, tol, max_iter, *params)
        - Bracket methods: func(f, a, b, tol, max_iter, *params)
        
        Where `*params` are 0 or more additional function parameters.
    num_params : int
        Number of function parameters. Must be >= 0. This determines how
        many parameter columns are expected in the func_params array.
        
        - num_params=0: No parameters (func_params will be empty array)
        - num_params=1: One parameter per solve
        - num_params=N: N parameters per solve
    method_type : MethodType or str
        Type of solver method template to generate. Determines the signature
        and structure of the generated vectorised solver.
        
        - MethodType.OPEN or 'open': For open methods (Newton, Secant)
          Requires x0 (initial guess)
        - MethodType.BRACKET or 'bracket': For bracket methods (Brent, Bisection)
          Requires a and b (bracket bounds)
    
    Returns
    -------
    vectorised_solver : callable
        JIT-compiled vectorised solver function with signature:
        
        - Open methods: 
          vectorised_solver(func, func_prime, func_params, x0, tol, max_iter)
          -> (roots, iterations, converged)
          
        - Bracket methods:
          vectorised_solver(func, func_params, a, b, tol, max_iter)
          -> (roots, iterations, converged)
        
        Where:
        - func: User's function to solve
        - func_prime: Derivative (open methods only)
        - func_params: ndarray, shape (n_solves, num_params)
        - x0/a/b: ndarray, shape (n_solves,)
        - roots: ndarray, shape (n_solves,) - Root locations
        - iterations: ndarray, shape (n_solves,) - Iteration counts
        - converged: ndarray, shape (n_solves,) - Convergence flags
    
    Raises
    ------
    ValueError
        If method_type is not MethodType.OPEN or MethodType.BRACKET.
    """

    scalar_name = scalar_func.__name__

    method_type = parse_enum(method_type, MethodType)

    if num_params > 0:
        param_names = ", ".join([f"p{i}" for i in range(num_params)])
        param_extracts = ", ".join([f"func_params[:, {i}]" for i in range(num_params)])
        param_indices = ", ".join([f"p{i}[i]" for i in range(num_params)])
        param_declaration = f"{param_names} = {param_extracts}"
        param_call = f", {param_indices}"

    else:
        param_declaration = ""
        param_call = ""

    if method_type == MethodType.OPEN:
        code = f"""

@njit(parallel = True)
def _open_solver_generated(func, func_prime, func_params, x0, tol, max_iter):

    n = len(x0)
    root_arr = np.empty(n, dtype=np.float64)
    iterations_arr = np.empty(n, dtype=np.int64)
    converged_flag_arr = np.empty(n, dtype=np.bool_)

    {param_declaration}

    for i in prange(n):
        root, iteration, converged_flag = {scalar_name}(
            func, func_prime, x0[i], tol, max_iter
            {param_call}
        )
        root_arr[i] = root
        iterations_arr[i] = iteration
        converged_flag_arr[i] = converged_flag

    return root_arr, iterations_arr, converged_flag_arr
"""
        func_name = "_open_solver_generated"

    elif method_type == MethodType.BRACKET:
        code = f"""

@njit(parallel = True)
def _bracket_solver_generated(func, func_params, a, b, tol, max_iter):

    n = len(a)
    root_arr = np.empty(n, dtype=np.float64)
    iterations_arr = np.empty(n, dtype=np.int64)
    converged_flag_arr = np.empty(n, dtype=np.bool_)

    {param_declaration}

    for i in prange(n):
        root, iteration, converged_flag = {scalar_name}(
            func, a[i], b[i], tol, max_iter
            {param_call}
        )
        root_arr[i] = root
        iterations_arr[i] = iteration
        converged_flag_arr[i] = converged_flag

    return root_arr, iterations_arr, converged_flag_arr
"""
        func_name = "_bracket_solver_generated"

    else:
        raise ValueError(f"Unsupported method type: {method_type.value}")

    namespace = {"njit": njit, "prange": prange, "np": np, scalar_name: scalar_func}

    exec(code, namespace)
    return namespace[func_name]
