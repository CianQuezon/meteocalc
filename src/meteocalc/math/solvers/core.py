"""
Core interface for Solvers

Author: Cian Quezon
"""

import numpy as np
import numpy.typing as npt
import warnings

from typing import List, Callable, Optional, Union, Tuple
from meteorological_equations.shared._enum_tools import parse_enum
from meteorological_equations.math.solvers._enums import SolverName, MethodType
from meteorological_equations.math.solvers._solvers import (
    BisectionSolver,
    NewtonRaphsonSolver,
    BrentSolver,
    Solver,
)

SolverMap = {
    SolverName.NEWTON: NewtonRaphsonSolver,
    SolverName.BISECTION: BisectionSolver,
    SolverName.BRENT: BrentSolver,
}


class RootSolvers:
    @staticmethod
    def list_root_solvers() -> List[str]:
        """
        Lists the solvers available
        """
        return [solver.value for solver in SolverName]

    @staticmethod
    def get_root(
        func: Callable[[float], float],
        a: Optional[Union[float, npt.ArrayLike]] = None,
        b: Optional[Union[float, npt.ArrayLike]] = None,
        x0: Optional[Union[float, npt.ArrayLike]] = None,
        func_prime: Optional[Callable[[float], float]] = None,
        func_params: Optional[npt.ArrayLike] = None,
        tol: float = 1e-6,
        max_iter=100,
        solver: Union[SolverName, str] = SolverName.NEWTON,
        use_backup: bool = True,
        backup_solvers: Optional[List[Union[str, SolverName]]] = [
            SolverName.BRENT,
            SolverName.BISECTION,
        ],
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Gets the roots of a function

        Args:
        """
        solver = parse_enum(solver, SolverName)
        solver_function = SolverMap[solver]

        method_type = RootSolvers._get_method_type(a=a, b=b, x0=0, solver=solver)
        
        if method_type == MethodType.OPEN:
            return solver_function(func=func, func_prime=func_prime, x0=x0, func_params=func_params, tol=tol, max_iter=max_iter)
        
        if method_type == MethodType.BRACKET:
            return solver_function(func=func, a=a, b=b, func_params=func_params, tol=tol, max_iter=max_iter)

    @staticmethod
    def _get_method_type(
        a: Optional[Union[float, npt.ArrayLike]],
        b: Optional[Union[float, npt.ArrayLike]],
        x0: Optional[Union[float, npt.ArrayLike]],
        solver: Optional[Solver] = None,
    ) -> MethodType:
        """
        Gets the method type of the input based on the inputs
        """
        if (a and b) and not x0:
            return MethodType.BRACKET

        if not (a and b) and x0:
            return MethodType.OPEN
        
    @staticmethod
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
        roots, iterations, converged_flag = results

        unconverged_mask = not(converged_flag)
        unconverged_idx = np.where(unconverged_mask)[0]
        
        if np.all(converged_flag):
            return results
        
        for backup_solver_name in backup_solvers:
            
            backup_solver_enum = parse_enum(backup_solver_name, SolverName)        
            back_up_solver_class = SolverMap[backup_solver_enum]
            back_up_solver = back_up_solver_class()

            method_type = back_up_solver.get_method_type()

            if method_type == (MethodType.OPEN or MethodType.HYBRID):

                try:
                    x0 = np.asarray(x0, dtype=np.float64)
                    x0_unconverged = x0[unconverged_idx]
                    func_params_unconverged = func_params[unconverged_idx]
                    
                    updated_roots, updated_iterations, updated_converged_flag = back_up_solver.find_root(
                        func=func,
                        func_prime=func_prime,
                        func_params=func_params_unconverged,
                        x0=x0_unconverged,
                        tol=tol
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

    def __try_back_up_scalar(
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

        for backup_solver_name in backup_solvers:
            back_up_solver










if __name__ == "__main__":
    print(RootSolvers.list_root_solvers())
