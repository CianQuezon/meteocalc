"""
Logic for using backup solvers.

Author: Cian Quezon
"""

import warnings
import numpy as np
import numpy.typing as npt

from typing import Callable, Optional, Union, Tuple, List
from meteorological_equations.math.solvers._solvers import Solver
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


def _try_back_up_vectorised(

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
    Docstring for __try_back_up_vectorised
    
    :param func: Description
    :type func: Callable[[float], float]
    :param results: Description
    :type results: Union[Tuple[float, int, bool], Tuple[npt.NDArray, npt.NDArray, npt.NDArray]]
    :param a: Description
    :type a: Optional[Union[npt.ArrayLike, float]]
    :param b: Description
    :type b: Optional[Union[npt.ArrayLike, float]]
    :param x0: Description
    :type x0: Optional[Union[npt.ArrayLike, float]]
    :param tol: Description
    :type tol: float
    :param max_iter: Description
    :type max_iter: int
    :param func_prime: Description
    :type func_prime: Optional[Callable[[float], float]]
    :param func_params: Description
    :type func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]]
    :param backup_solvers: Description
    :type backup_solvers: List[Union[str, MethodType]]
    """
    roots, iterations, converged_flag = results

    unconverged_mask = not converged_flag
    unconverged_idx = np.where(unconverged_mask)[0]
    
    if np.all(converged_flag):
        return results
    
    for backup_solver_name in backup_solvers:
        
        backup_solver_enum = parse_enum(backup_solver_name, SolverName)        
        back_up_solver_class = SolverMap[backup_solver_enum]
        back_up_solver = back_up_solver_class()

        method_type = back_up_solver.get_method_type()

        if method_type == MethodType.HYBRID:

            if x0 is not None:
                
                try:
                    x0 = np.asarray(x0, dtype=np.float64)
                    
                    x0_unconverged = x0[unconverged_idx]
                    
                    
                    if func_params is not None:
                        func_params = np.asarray(func_params, dtype=np.float64)
                        func_params_unconverged = func_params[unconverged_idx]
                    
                    else:
                        func_params_unconverged = None

                    updated_roots, updated_iterations, updated_converged_flag = back_up_solver.find_root(
                        func=func,
                        func_prime=func_prime,
                        func_params=func_params_unconverged,
                        x0=x0_unconverged,
                        tol=tol,
                        max_iter=max_iter
                    )

                    newly_converged_mask = updated_converged_flag
                    newly_converged_original_idx = unconverged_idx[newly_converged_mask]

                    roots[newly_converged_original_idx] = updated_roots[newly_converged_mask]
                    iterations[newly_converged_original_idx] = updated_iterations[newly_converged_mask]
                    converged_flag[newly_converged_original_idx] = True
                
                except Exception as e:
                    warnings.warn(f"{backup_solver_enum.value} did not converge. Skipping to the next solver")

                

        if method_type == MethodType.OPEN:

            try:
                x0 = np.asarray(x0, dtype=np.float64)
                x0_unconverged = x0[unconverged_idx]
                func_params_unconverged = func_params[unconverged_idx]
                
                updated_roots, updated_iterations, updated_converged_flag = back_up_solver.find_root(
                    func=func,
                    func_prime=func_prime,
                    func_params=func_params_unconverged,
                    x0=x0_unconverged,
                    tol=tol,
                    max_iter=max_iter
                )
            
            except:
                if method_type == MethodType.OPEN:

                    pass
                continue

            results[unconverged_idx] = (updated_roots, updated_iterations, updated_converged_flag)
            
        if method_type == (MethodType.HYBRID or MethodType.BRACKET):
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            func_params_unconverged = func_params[unconverged_idx]

            updated_roots, updated_iterations, updated_converged_flag = back_up_solver.find_root(
                func=func,
                a=a,
                b=b,
                func_params=func_params,
                tol=tol,
                max_iter=max_iter
            )

        
        results[unconverged_idx] = (updated_roots, updated_iterations, updated_converged_flag)
        unconverged_mask = not(converged_flag)
        unconverged_idx = np.where(unconverged_mask)[0]





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


def _try_back_up_bracket_vectorised(
    back_up_solver: Solver,
    func: Callable[[float], float],
    results: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    unconverged_idx: npt.ArrayLike,
    tol: float,
    max_iter: int,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]],
        
) -> bool:
    """
    Docstring for __try_back_up_bracket
    
    """

    try: 
        original_roots = results[0]
        original_iterations = results[1]
        original_converged_flag = results[2]

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        a_unconverged = a[unconverged_idx]
        b_unconverged = b[unconverged_idx]

        func_params_unconverged = _get_unconverged_func_params(
            func_params=func_params, unconverged_idx=unconverged_idx
        )

        updated_roots, updated_iterations, updated_converged_flag = back_up_solver.find_root(
            func=func,
            a=a_unconverged,
            b=b_unconverged,
            func_params=func_params_unconverged,
            tol=tol,
            max_iter=max_iter
        )

        _update_converged_results(
            roots=original_roots,
            iterations=original_iterations,
            converged_flag=original_converged_flag,
            unconverged_idx=unconverged_idx,
            updated_roots=updated_roots,
            updated_iterations=updated_iterations,
            updated_converged_flag=updated_converged_flag
        )

        return np.any(updated_converged_flag)

    except Exception as e:             
        warnings.warn(f"Bracketing method failed: {e}. Skipping to the next solver.")
        return False


def _try_back_up_open_vectorised(
    back_up_solver: Solver,
    func: Callable[[float], float],
    results: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    x0: npt.ArrayLike,
    unconverged_idx: npt.ArrayLike,
    tol: float,
    max_iter: int,
    func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]],
    func_prime: Optional[Callable[[float], float]]
) -> bool:
    """
    Docstring for _try_back_up_open_vectorised

    """

    try:

        original_roots = results[0]
        original_iterations = results[1]
        original_converged_flag = results[2]

        x0 = np.asarray(x0, dtype=np.float64)
        x0_unconverged = x0[unconverged_idx]

        func_params_unconverged = _get_unconverged_func_params(
            func_params=func_params,
            unconverged_idx=unconverged_idx
        ) 

        updated_roots, updated_iterations, updated_converged_flag = back_up_solver.find_root(
            func=func,
            func_prime=func_prime,
            x0=x0_unconverged,
            func_params=func_params_unconverged,
            tol=tol,
            max_iter=max_iter
        )

        _update_converged_results(
            roots=original_roots,
            iterations=original_iterations,
            converged_flag=original_converged_flag,
            unconverged_idx=unconverged_idx,
            updated_roots=updated_roots,
            updated_iterations=updated_iterations,
            updated_converged_flag=updated_converged_flag
        )

        return np.any(updated_converged_flag)

    except Exception as e:
        warnings.warn(f"Open method failed: {e}")
        return False


def _get_unconverged_func_params(
        func_params: Optional[Union[npt.ArrayLike, Tuple[float, ...]]],
        unconverged_idx: npt.ArrayLike
) -> Optional[npt.NDArray]:
    """
    Extracts the unconverged function parameters if not None.

    Args:
        - func_params (Optional[Union[npt.ArrayLike, Tuple[float, ...]]]) = function parameters which is either None or array of tuples
        - unconverged_idx = position of function params that have not converged
    
    
    Returns:
        - function parameters or None 

    """
    if func_params is not None:
        func_params = np.asarray(func_params, dtype=np.float64)
        return func_params[unconverged_idx]
    
    return None

def _update_converged_results(
    roots: npt.NDArray,
    iterations: npt.NDArray,
    converged_flag: npt.NDArray,
    unconverged_idx: npt.NDArray,
    updated_roots: npt.NDArray,
    updated_iterations: npt.NDArray,
    updated_converged_flag: npt.NDArray
) -> None:
    """
    Updates only the positions that have converged

    Args:
        - root
    """
    updated_converged_mask = updated_converged_flag
    updated_converged_original_idx = unconverged_idx[updated_converged_mask]
    
    
    roots[updated_converged_original_idx] = updated_roots[updated_converged_mask]
    iterations[updated_converged_original_idx] = updated_iterations[updated_converged_mask]
    converged_flag[updated_converged_original_idx] = True
