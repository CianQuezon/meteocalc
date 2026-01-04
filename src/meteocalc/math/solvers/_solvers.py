"""
A interface for JIT solvers.

Author: Cian Quezon
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from meteorological_equations.math.solvers._enums import MethodType, SolverName
from meteorological_equations.math.solvers._jit_solvers import (
    _bisection_scalar,
    _bisection_vectorised,
    _brent_scalar,
    _brent_vectorised,
    _newton_raphson_scalar,
    _newton_raphson_vectorised,
)

BracketRootMethodScalar = Callable[
    [Callable[[float], float], float, float, float, int], Tuple[float, int, bool]
]

BracketRootMethodVectorised = Callable[
    [Callable[[float], float], npt.ArrayLike, npt.ArrayLike, float, int],
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
]


OpenRootMethodScalar = Callable[
    [Callable[[float], float], Optional[Callable[[float], float]], float, int],
    Tuple[float, int, bool],
]

OpenRootMethodVectorised = Callable[
    [Callable[[float], float], Optional[Callable[[float], float]], npt.ArrayLike, int],
    Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
]


class Solver(ABC):
    """
    Base abstract class for solvers.

    Attributes:
        name (SolverName) =
    """

    method_type: MethodType
    name: SolverName

    def _dispatch_root_bracket_method(
        self,
        func: Callable[[float], float],
        a: Union[float, npt.ArrayLike],
        b: Union[float, npt.ArrayLike],
        scalar_bracket_method_func: BracketRootMethodScalar,
        vector_bracket_method_func: BracketRootMethodVectorised,
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Dispatches the bracket root method and chooses scalar or vectorised versions based on input.

        Args:
            - func (Callable[[float], float]) = function required to solve the root
            - a (Union[float, npt.Arraylike[float]]) = an array or float of the lower bracket bound
            - b (Union[float, npt.Arraylike[float]]) = an array or float of the upper bracket bound
            - scalar_bracket_method_func(BracketRootMethodScalar) = solver function for dealing with scalars
            - vector_bracket_method_func(BracketRootMethodVectorised) = solver function for dealing with vectors
            - tol(float) = Tolerance for convergence
            - max_iter (int) = Maximum iterations

        Returns:
            - an array or scalar of (root, iterations, converged)
        """
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        if a_arr.ndim == 0:
            return scalar_bracket_method_func(func=func, a=a, b=b, tol=tol, max_iter=max_iter)

        else:
            original_shape = a_arr.shape
            a_flatten = a_arr.flatten()
            b_flatten = b_arr.flatten()

            roots, iterations, converged_flags = vector_bracket_method_func(
                func=func, a=a_flatten, b=b_flatten, tol=tol, max_iter=max_iter
            )

            roots = np.asarray(roots, dtype=np.float64)
            iterations = np.asarray(iterations, dtype=np.int64)
            converged_flags = np.asarray(converged_flags, dtype=np.bool_)

            return (
                roots.reshape(original_shape),
                iterations.reshape(original_shape),
                converged_flags.reshape(original_shape),
            )

    def _dispatch_root_open_method(
        self,
        func: Callable[[float], float],
        func_prime: Optional[Callable[[float], float]],
        x0: Union[npt.ArrayLike, float],
        scalar_open_method_func: OpenRootMethodScalar,
        vectorised_open_method_func: OpenRootMethodVectorised,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Dispatches the Open root method and chooses scalar or vectorised versions based on input.
        """
        initial_guess_arr = np.asarray(x0, dtype=np.float64)

        if initial_guess_arr.ndim == 0:
            if func_prime is not None:
                return scalar_open_method_func(
                    func=func, func_prime=func_prime, x0=x0, tol=tol, max_iter=max_iter
                )
            else:
                return scalar_open_method_func(func=func, x0=x0, tol=tol, max_iter=max_iter)

        else:
            original_shape = initial_guess_arr.shape
            x0_flatten = initial_guess_arr.flatten()

            if func_prime is not None:
                roots, iterations, converged_flags = vectorised_open_method_func(
                    func=func, func_prime=func_prime, x0=x0_flatten, tol=tol, max_iter=max_iter
                )
            else:
                roots, iterations, converged_flags = vectorised_open_method_func(
                    func=func, x0=x0_flatten, tol=tol, max_iter=max_iter
                )

            roots = np.asarray(roots, dtype=np.float64)
            iterations = np.asarray(iterations, dtype=np.int64)
            converged_flags = np.asarray(converged_flags, dtype=np.bool_)

            return (
                roots.reshape(original_shape),
                iterations.reshape(original_shape),
                converged_flags.reshape(original_shape),
            )

    @abstractmethod
    def find_root(self, *args, **kwargs):
        pass


class NewtonRaphsonSolver(Solver):
    """
    Newton Raphon Solver class to find roots for an equation
    """

    def __init__(self):
        """
        Initialises the solvers with the name and method type attributes
        """
        self.name = SolverName.NEWTON
        self.method_type = MethodType.OPEN

    def find_root(
        self,
        func: Callable[[float], float],
        func_prime: Callable[[float], float],
        x0: Union[float, npt.ArrayLike],
        tol: float = 1e-6,
        max_iter: int = 50,
    ):
        """
        Finds the roots of a function using the Newton Raphson Method.

        Args:
            - func(Callable[[float], float]) = function required to solve the root
            - func_prime(Callable[[float], float]) = derivative of the function
            - x0(Union[npt.ArrayLike[np.float64], float]) = Array or a scalar of initial guesses
            - tol (float) = Convergence tolerance
            - max_iter(int) = maximum amount of iterations

        Returns:
            Scalar or array of (root, iterations, converged)
        """
        return self._dispatch_root_open_method(
            func=func,
            func_prime=func_prime,
            x0=x0,
            scalar_open_method_func=_newton_raphson_scalar,
            vectorised_open_method_func=_newton_raphson_vectorised,
            tol=tol,
            max_iter=max_iter,
        )


class BisectionSolver(Solver):
    """
    Bisection Solver class to find roots for an equation
    """

    def __init__(self):
        self.name = SolverName.BISECTION
        self.method_type = MethodType.BRACKET

    def find_root(
        self,
        func: Callable[[float], float],
        a: Union[npt.ArrayLike, float],
        b: Union[npt.ArrayLike, float],
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        """
        Finds the roots of a function using the Bisection Method.

        Args:
            - func: Callable[[float], float] = Function to solve for the root
            - a (Union[npt.NDArray[float], float]) = Scalar or an array of Upper bracket bounds
            - b (Union[npt.NDArray[float], float]) = Scalar or an array of Lower bracket bound
            - tol (float) = Convergence tolerance
            - max_iter (int) = Maximum amount of iterations

        Returns:
            - Array or scalar of roots in (root, iterations, converged)
        """

        return self._dispatch_root_bracket_method(
            func=func,
            a=a,
            b=b,
            scalar_bracket_method_func=_bisection_scalar,
            vector_bracket_method_func=_bisection_vectorised,
            tol=tol,
            max_iter=max_iter,
        )


class BrentSolver(Solver):
    """
    Brent Solver class to find roots for an equation. It is a hybrid since it uses open methods internally but uses bracketing
    for convergence.
    """

    def __init__(self):
        self.name = SolverName.BRENT
        self.method_type = MethodType.HYBRID

    def find_root(
        self,
        func: Callable[[float], float],
        a: Union[npt.ArrayLike, float],
        b: Union[npt.ArrayLike, float],
        tol: float = 1e-6,
        max_iter: int = 100,
    ):
        """
        Finds the roots of a function using the Brent Method.

        Args:
            - func: Callable[[float], float] = Function to solve for the root
            - a (Union[npt.NDArray[float], float]) = Scalar or an array of Upper bracket bounds
            - b (Union[npt.NDArray[float], float]) = Scalar or an array of Lower bracket bound
            - tol (float) = Convergence tolerance
            - max_iter (int) = Maximum amount of iterations

        Returns:
            - Array or scalar of roots in (root, iterations, converged)
        """
        return self._dispatch_root_bracket_method(
            func=func,
            a=a,
            b=b,
            scalar_bracket_method_func=_brent_scalar,
            vector_bracket_method_func=_brent_vectorised,
            tol=tol,
            max_iter=max_iter,
        )
