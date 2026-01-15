"""
Logic for using backup solvers.

Author: Cian Quezon
"""

import warnings
import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Union, Tuple, List
from meteorological_equations.shared._enum_tools import parse_enum
from meteorological_equations.math.solvers._enums import MethodType, SolverName
from meteorological_equations.math.solvers.core import SolverMap



def _use_back_up_solvers(
    func: Callable[[float], float],
    results: Union[Tuple[float, int, bool], Tuple[npt.NDArray, npt.NDArray, npt.NDArray]],
    a: Optional[Union[npt.ArrayLike, float]],
    b: Optional[Union[npt.ArrayLike, float]],
    x0: Optional[Union[npt.ArrayLike, float]],
    tol: float,
    max_iter: int,
    func_prime: Optional[Callable[[float], float]] = None,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]] = None,
    backup_solvers: List[Union[str, MethodType]] = [SolverName.BRENT, SolverName.BISECTION]
) -> Union[
    Tuple[float, int, bool],
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
]:
    """
    Uses backup solvers for functions that has not converged if use_backups is true.
        
    Args:
        - a (Optional[Union[ArrayLike, float]]) = Array or a acalar of upper bound brackets
        - b (Optional[Union[ArrayLike, float]]) = Array or a scalar of lower bound brackets
        - x0 (Optional[Union[ArrayLike, float]]) = Array or a scalar of inital guesses 
        
    Returns:
        An array or a scalar of (float, int, bool)
    """     

    roots = np.asarray(results[0])
    iterations = np.asarray(results[1])
    converged_flag = np.asarray(results[2])

    if roots.ndim == 0:
        scalar_results = (float(roots), int(iterations), bool(converged_flag))
            
        scalar_a = float(a) if a is not None else None
        scalar_b = float(b) if b is not None else None
        scalar_x0 = float(x0) if x0 is not None else None



        return _try_back_up_scalar(
            func=func,
            results=scalar_results,
            a=scalar_a,
            b=scalar_b,
            x0=scalar_x0,
            tol=tol,
            max_iter=max_iter,
            func_prime=func_prime,
            func_params=func_params,
            backup_solvers=backup_solvers

        )
        
    else:
        return results





def _try_back_up_scalar(
    func: Callable[[float], float],
    results: Union[Tuple[float, int, bool], Tuple[npt.NDArray, npt.NDArray, npt.NDArray]],
    a: Optional[Union[npt.ArrayLike, float]],
    b: Optional[Union[npt.ArrayLike, float]],
    x0: Optional[Union[npt.ArrayLike, float]],
    tol: float,
    max_iter: int,
    func_prime: Optional[Callable[[float], float]] = None,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]] = None,
    backup_solvers: List[Union[str, MethodType]] = [SolverName.BRENT, SolverName.BISECTION],
            
    ):
    """
    Docstring for __dispatch_back_up_scalar

    """
    converged_flag = results[2]

    if converged_flag:
        return results

    for backup_solver_name in backup_solvers:
            
        back_up_solver_enum = parse_enum(backup_solver_name, SolverName)
        back_up_solver = SolverMap[back_up_solver_enum]()

        method_type = back_up_solver.get_method_type()
            
        if method_type == MethodType.HYBRID:
            converged_flag = False
                
            if x0 is not None:
                    
                try:
                    results = back_up_solver.find_root(func=func, func_prime=func_prime, x0=x0, func_params=func_params,
                                                    tol=tol, max_iter=max_iter)
                    converged_flag = results[2]
                        
                    if converged_flag:
                        return results
                        
                except Exception as e:
                    warnings.warn(f"Open method failed: {e}")

            if a is not None and b is not None:
                    
                try:
                    results = back_up_solver.find_root(func=func, a=a, b=b, func_params=func_params,
                                                        tol=tol, max_iter=max_iter)
                    converged_flag = results[2]
                    if converged_flag:
                        return results
                        
                except Exception as e:
                    warnings.warn(f"Bracketing method failed: {e}")

                
            if not converged_flag:
                warnings.warn(f"{back_up_solver_enum.value} did not converge. Skipping to the next solver")
                continue


        elif method_type == MethodType.BRACKET:

            try:
                results = back_up_solver.find_root(func=func, a=a, b=b, func_params=func_params,
                                                        tol=tol, max_iter=max_iter)
                    
                converged_flag = results[2]

                if not converged_flag:
                    warnings.warn(f"{back_up_solver_enum.value} did not converge. Skipping to the next solver")
                    continue

                return results

            except Exception as e:
                     
                warnings.warn(f"Bracketing method failed: {e}. Skipping to the next solver.")
                continue
            
        elif method_type == MethodType.OPEN:
                
            try:
                results = back_up_solver.find_root(func=func, func_prime=func_prime, x0=x0, func_params=func_params,
                                                        tol=tol, max_iter=max_iter)
                converged_flag = results[2]

                if not converged_flag:
                    warnings.warn(f"{back_up_solver_enum.value} did not converge. Skipping to the next solver")
                    continue
                return results
                
            except Exception as e:
                     
                warnings.warn(f"Open method failed: {e}. Skipping to the next solver.")
                continue
            
        else:
            warnings.warn("Unknown method type. Skipping to the next solver")
        
    return results
