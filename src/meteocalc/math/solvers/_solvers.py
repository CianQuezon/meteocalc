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
    Abstract base class for all root-finding solvers.
    
    This class defines the common interface and shared functionality for all
    solver implementations. It provides automatic dispatching between scalar
    and vectorised execution paths, shape preservation for multi-dimensional
    inputs, and parameter preparation utilities.
    
    Concrete solver classes (NewtonRaphsonSolver, BisectionSolver, BrentSolver)
    inherit from this base and implement the `find_root` method with their
    specific algorithm requirements.
    
    Attributes
    ----------
    method_type : MethodType
        Classification of the solver method (OPEN, BRACKET, HYBRID, CUSTOM).
        Set by concrete subclasses in __init__.
    name : SolverName
        Unique identifier for the solver algorithm (NEWTON, BRENT, BISECTION).
        Set by concrete subclasses in __init__.
    
    See Also
    --------
    NewtonRaphsonSolver : Newton-Raphson method implementation
    BisectionSolver : Bisection method implementation
    BrentSolver : Brent's method implementation
    """

    method_type: MethodType
    name: SolverName

    def get_method_type(self) -> str:
        """
        Get the method type classification of this solver.
        
        Returns the MethodType enum indicating whether this is an open method
        (requires initial guess), bracket method (requires interval bounds),
        hybrid method (combines both approaches), or custom method.
        
        Returns
        -------
        MethodType
            Classification of this solver's algorithmic approach:
            - MethodType.OPEN: Newton-Raphson, Secant
            - MethodType.BRACKET: Bisection, Regula Falsi
            - MethodType.HYBRID: Brent (uses both bracketing and interpolation)
            - MethodType.CUSTOM: User-defined methods
        
        See Also
        --------
        MethodType : Enumeration of solver method categories
        """
        return self.method_type

    @staticmethod
    def _prepare_scalar_params(
        func_params: Optional[Union[Tuple[float, ...], npt.ArrayLike]],
    ) -> Tuple[float, ...]:
        """
        Prepare function parameters for scalar solver execution.
        
        Converts various parameter input formats into a tuple that can be
        unpacked with the * operator for passing to JIT-compiled scalar
        functions. This ensures compatibility with Numba's requirement for
        explicit parameter passing.
        
        Parameters
        ----------
        func_params : tuple, array_like, or None
            Function parameters in any of these formats:
            - None: No parameters (returns empty tuple)
            - tuple: Already in correct format (returns as-is)
            - list or ndarray: Converted to flattened tuple
            - scalar: Wrapped in single-element tuple
        
        Returns
        -------
        tuple of float
            Parameters as tuple ready for unpacking with * operator.
            Empty tuple () if func_params is None.
        
        See Also
        --------
        _dispatch_root_bracket_method : Uses this for parameter preparation
        _dispatch_root_open_method : Uses this for parameter preparation
        """

        if func_params is None:
            return ()

        if isinstance(func_params, tuple):
            return func_params

        if isinstance(func_params, (list, np.ndarray)):
            arr = np.asarray(func_params).flatten()
            return tuple(arr)

        return func_params

    def _dispatch_root_bracket_method(
        self,
        func: Callable[[float], float],
        a: Union[float, npt.ArrayLike],
        b: Union[float, npt.ArrayLike],
        scalar_bracket_method_func: BracketRootMethodScalar,
        vector_bracket_method_func: BracketRootMethodVectorised,
        func_params: Union[Optional[npt.ArrayLike], Tuple[float, ...]] = None,
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Dispatch bracket methods to scalar or vectorised implementations.
        
        This internal dispatcher automatically selects between scalar and
        vectorised execution paths based on input types, handles multi-dimensional
        array shapes, and ensures consistent output formatting.
        
        Used by bracket-based solvers (Bisection, Brent) to provide a unified
        interface regardless of input type.
        
        Parameters
        ----------
        func : callable
            Function for which to find roots. Should be JIT-compiled.
        a : float or array_like
            Lower bracket bound(s). Scalar for single problem, array for
            multiple problems.
        b : float or array_like
            Upper bracket bound(s). Must have same shape as `a`.
        scalar_bracket_method_func : callable
            JIT-compiled scalar solver function (e.g., _bisection_scalar).
        vector_bracket_method_func : callable
            JIT-compiled vectorised solver function (e.g., _bisection_vectorised).
        func_params : array_like, tuple, or None, optional
            Additional parameters for func. Format depends on scalar vs vectorised.
        tol : float, default=1e-6
            Convergence tolerance.
        max_iter : int, default=50
            Maximum iterations per problem.
        
        Returns
        -------
        scalar case (a is 0D scalar):
            root : float
                Root location
            iterations : int
                Number of iterations performed
            converged : bool
                Convergence flag
        
        vectorised case (a is array):
            roots : ndarray, same shape as input `a`
                Root locations
            iterations : ndarray, same shape as input `a`
                Iteration counts per problem
            converged : ndarray, same shape as input `a`
                Convergence flags per problem

        See Also
        --------
        _dispatch_root_open_method : Similar dispatcher for open methods
        BisectionSolver.find_root : Uses this dispatcher
        BrentSolver.find_root : Uses this dispatcher
        """
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        if a_arr.ndim == 0:
            params = self._prepare_scalar_params(func_params=func_params)
            return scalar_bracket_method_func(func, a, b, tol, max_iter, *params)

        else:
            original_shape = a_arr.shape
            a_flatten = a_arr.flatten()
            b_flatten = b_arr.flatten()

            roots, iterations, converged_flags = vector_bracket_method_func(
                func=func,
                a=a_flatten,
                b=b_flatten,
                func_params=func_params,
                tol=tol,
                max_iter=max_iter,
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
        func_params: Optional[Union[npt.ArrayLike, Tuple[float, ...]]] = None,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Dispatch open methods to scalar or vectorised implementations.
        
        This internal dispatcher automatically selects between scalar and
        vectorised execution paths for open methods (Newton-Raphson, Secant),
        handles multi-dimensional array shapes, and manages optional derivative
        functions.
        
        Parameters
        ----------
        func : callable
            Function for which to find roots. Should be JIT-compiled.
        func_prime : callable or None
            Derivative of func. Required for Newton's method, optional for
            other open methods.
        x0 : float or array_like
            Initial guess(es). Scalar for single problem, array for multiple.
        scalar_open_method_func : callable
            JIT-compiled scalar solver function (e.g., _newton_raphson_scalar).
        vectorised_open_method_func : callable
            JIT-compiled vectorised solver function (e.g., _newton_raphson_vectorised).
        func_params : array_like, tuple, or None, optional
            Additional parameters for func and func_prime.
        tol : float, default=1e-6
            Convergence tolerance.
        max_iter : int, default=100
            Maximum iterations per problem.
        
        Returns
        -------
        scalar case (x0 is 0D scalar):
            root : float
                Root location
            iterations : int
                Number of iterations performed
            converged : bool
                Convergence flag
        
        vectorised case (x0 is array):
            roots : ndarray, same shape as input `x0`
                Root locations
            iterations : ndarray, same shape as input `x0`
                Iteration counts per problem
            converged : ndarray, same shape as input `x0`
                Convergence flags per problem

        See Also
        --------
        _dispatch_root_bracket_method : Similar dispatcher for bracket methods
        NewtonRaphsonSolver.find_root : Uses this dispatcher
        """
        initial_guess_arr = np.asarray(x0, dtype=np.float64)

        if initial_guess_arr.ndim == 0:
            params = self._prepare_scalar_params(func_params=func_params)
            if func_prime is not None:
                return scalar_open_method_func(func, func_prime, x0, tol, max_iter, *params)
            else:
                return scalar_open_method_func(func, x0, tol, max_iter, *params)

        else:
            original_shape = initial_guess_arr.shape
            x0_flatten = initial_guess_arr.flatten()

            if func_prime is not None:
                roots, iterations, converged_flags = vectorised_open_method_func(
                    func=func,
                    func_prime=func_prime,
                    x0=x0_flatten,
                    func_params=func_params,
                    tol=tol,
                    max_iter=max_iter,
                )
            else:
                roots, iterations, converged_flags = vectorised_open_method_func(
                    func=func, x0=x0_flatten, func_params=func_params, tol=tol, max_iter=max_iter
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
        """
        Find roots of a function (abstract method).
        
        This method must be implemented by all concrete solver subclasses.
        The specific signature and requirements depend on the solver type
        (open vs bracket methods).
        
        Raises
        ------
        NotImplementedError
            If called on the base Solver class directly.
        
        See Also
        --------
        NewtonRaphsonSolver.find_root : Open method implementation
        BisectionSolver.find_root : Bracket method implementation
        BrentSolver.find_root : Hybrid method implementation
        """
        pass


class NewtonRaphsonSolver(Solver):
        """
        Find roots of a function (abstract method).
        
        This method must be implemented by all concrete solver subclasses.
        The specific signature and requirements depend on the solver type
        (open vs bracket methods).
        
        Raises
        ------
        NotImplementedError
            If called on the base Solver class directly.
        
        See Also
        --------
        NewtonRaphsonSolver.find_root : Open method implementation
        BisectionSolver.find_root : Bracket method implementation
        BrentSolver.find_root : Hybrid method implementation
        """
        pass


class NewtonRaphsonSolver(Solver):
    """
    Newton-Raphson method solver for root finding.
    
    Implements Newton's method, which uses the function and its derivative
    to iteratively refine an initial guess. Provides fast quadratic convergence
    when the initial guess is good and the function is well-behaved.
    
    The solver automatically dispatches between scalar and vectorised execution
    based on input types, supporting both single and batch root-finding problems.
    
    Attributes
    ----------
    name : SolverName
        Set to SolverName.NEWTON
    method_type : MethodType
        Set to MethodType.OPEN
    
    Examples
    --------
    Scalar root finding:
    
    >>> from numba import njit
    >>> 
    >>> @njit
    ... def func(x):
    ...     return x**2 - 4
    >>> 
    >>> @njit
    ... def func_prime(x):
    ...     return 2*x
    >>> 
    >>> solver = NewtonRaphsonSolver()
    >>> root, iters, conv = solver.find_root(
    ...     func, func_prime, x0=1.0, tol=1e-6, max_iter=50
    ... )
    >>> print(f"Root: {root:.6f}, Iterations: {iters}")
    Root: 2.000000, Iterations: 4
    
    Vectorised root finding:
    
    >>> import numpy as np
    >>> x0 = np.array([1.0, 1.5, 0.5])
    >>> roots, iters, conv = solver.find_root(
    ...     func, func_prime, x0=x0, tol=1e-6
    ... )
    >>> print("Roots:", roots)
    Roots: [2.0, 2.0, 2.0]
    >>> print("Iterations:", iters)
    Iterations: [4, 3, 5]
    
    With function parameters:
    
    >>> @njit
    ... def parametric_func(x, a, b):
    ...     return a * x**2 + b
    >>> 
    >>> @njit
    ... def parametric_prime(x, a, b):
    ...     return 2 * a * x
    >>> 
    >>> # Different parameters per problem
    >>> func_params = np.array([[1.0, -4.0], [2.0, -8.0]])
    >>> roots, iters, conv = solver.find_root(
    ...     parametric_func, parametric_prime,
    ...     x0=np.array([1.0, 1.0]),
    ...     func_params=func_params
    ... )
    
    See Also
    --------
    BrentSolver : More robust alternative (no derivative needed)
    BisectionSolver : Guaranteed convergence (no derivative needed)
    _newton_raphson_scalar : Underlying scalar implementation
    _newton_raphson_vectorised : Underlying vectorised implementation
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
        func_params: Optional[Union[Tuple[float, ...], npt.ArrayLike]] = None,
        tol: float = 1e-6,
        max_iter: int = 50,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Find roots using Newton-Raphson method.
        
        Iteratively refines an initial guess using Newton's formula:
        x_{n+1} = x_n - f(x_n) / f'(x_n)
        
        Automatically dispatches to scalar or vectorised implementation based
        on x0 type. Supports multi-dimensional arrays with shape preservation.
        
        Parameters
        ----------
        func : callable
            Function for which to find roots. Must be JIT-compiled (@njit)
            and have signature: func(x, *params) -> float.
        func_prime : callable
            Derivative of func with respect to x. Must be JIT-compiled and
            have same signature as func. Accuracy is critical for convergence.
        x0 : float or array_like
            Initial guess(es) for root location(s). Quality significantly
            affects convergence rate and success.
            - Scalar: Single root-finding problem
            - Array: Multiple independent problems (vectorised)
        func_params : tuple, array_like, or None, optional
            Additional parameters for func and func_prime.
            - None: No parameters
            - Tuple/list: Same parameters for all problems
            - 2D array: Different parameters per problem (vectorised)
        tol : float, default=1e-6
            Convergence tolerance. Stops when |x_{n+1} - x_n| < tol.
        max_iter : int, default=50
            Maximum iterations allowed per problem. Newton typically
            converges in 3-8 iterations with good x0.
        
        Returns
        -------
        scalar case (x0 is scalar):
            root : float
                Root location (NaN if unconverged)
            iterations : int
                Number of iterations performed [0, max_iter]
            converged : bool
                True if tolerance met, False if failed
        
        vectorised case (x0 is array):
            roots : ndarray, same shape as x0
                Root locations for each problem
            iterations : ndarray, same shape as x0
                Iteration counts per problem
            converged : ndarray, same shape as x0
                Convergence flags per problem
        
        Raises
        ------
        TypeError
            If func or func_prime are not callable
        ValueError
            If x0, func_params shapes incompatible
        
        See Also
        --------
        BrentSolver.find_root : Alternative without derivative
        _newton_raphson_scalar : Scalar implementation details
        _newton_raphson_vectorised : Vectorised implementation details
        
        Examples
        --------
        See class docstring for comprehensive examples.
        """
        return self._dispatch_root_open_method(
            func=func,
            func_prime=func_prime,
            x0=x0,
            scalar_open_method_func=_newton_raphson_scalar,
            vectorised_open_method_func=_newton_raphson_vectorised,
            func_params=func_params,
            tol=tol,
            max_iter=max_iter,
        )


class BisectionSolver(Solver):
    """
    Bisection method solver for root finding.
    
    Implements the bisection algorithm, which repeatedly halves an interval
    where the function changes sign. Provides guaranteed linear convergence
    for continuous functions with valid brackets.
    
    The solver is the most robust root-finding method but slower than
    Newton or Brent. It automatically dispatches between scalar and vectorised
    execution based on input types.
    
    Attributes
    ----------
    name : SolverName
        Set to SolverName.BISECTION
    method_type : MethodType
        Set to MethodType.BRACKET
    
    Examples
    --------
    Scalar root finding:
    
    >>> from numba import njit
    >>> 
    >>> @njit
    ... def func(x):
    ...     return x**2 - 4
    >>> 
    >>> solver = BisectionSolver()
    >>> root, iters, conv = solver.find_root(
    ...     func, a=0.0, b=5.0, tol=1e-6, max_iter=100
    ... )
    >>> print(f"Root: {root:.6f}, Iterations: {iters}")
    Root: 2.000000, Iterations: 23
    
    Vectorised root finding:
    
    >>> import numpy as np
    >>> a = np.array([0.0, -5.0, 1.0])
    >>> b = np.array([5.0, 0.0, 3.0])
    >>> roots, iters, conv = solver.find_root(
    ...     func, a=a, b=b, tol=1e-6
    ... )
    >>> print("Roots:", roots)
    Roots: [2.0, -2.0, 2.0]
    
    Invalid bracket (no sign change):
    
    >>> # Both endpoints positive - returns NaN
    >>> root, iters, conv = solver.find_root(
    ...     func, a=3.0, b=5.0
    ... )
    >>> print(f"Converged: {conv}, Root: {root}")
    Converged: False, Root: nan
    
    See Also
    --------
    BrentSolver : Faster alternative with same robustness (recommended)
    NewtonRaphsonSolver : Much faster but requires derivative
    _bisection_scalar : Underlying scalar implementation
    _bisection_vectorised : Underlying vectorised implementation
    """
    def __init__(self):
        self.name = SolverName.BISECTION
        self.method_type = MethodType.BRACKET

    def find_root(
        self,
        func: Callable[[float], float],
        a: Union[npt.ArrayLike, float],
        b: Union[npt.ArrayLike, float],
        func_params: Optional[Union[Tuple[float, ...], npt.ArrayLike]] = None,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Find roots using bisection method.
        
        Repeatedly halves the interval [a, b] where f(a) and f(b) have
        opposite signs until convergence or max_iter reached.
        
        Automatically dispatches to scalar or vectorised implementation.
        Supports multi-dimensional arrays with shape preservation.
        
        Parameters
        ----------
        func : callable
            Function for which to find roots. Must be JIT-compiled (@njit)
            and have signature: func(x, *params) -> float.
        a : float or array_like
            Lower bracket bound(s). Must satisfy f(a) * f(b) < 0 for
            valid bracket (sign change required).
        b : float or array_like
            Upper bracket bound(s). Must have same shape as `a` and
            satisfy f(a) * f(b) < 0.
        func_params : tuple, array_like, or None, optional
            Additional parameters for func.
        tol : float, default=1e-6
            Convergence tolerance. Stops when |f(c)| < tol or
            |b-a|/2 < tol where c is midpoint.
        max_iter : int, default=100
            Maximum iterations allowed per problem. Bisection typically
            needs ≈ log₂((b-a)/tol) iterations ≈ 20-25.
        
        Returns
        -------
        scalar case (a is scalar):
            root : float
                Root location (NaN if invalid bracket)
            iterations : int
                Number of iterations [0, max_iter]
            converged : bool
                True if converged, False if invalid bracket or max_iter
        
        vectorised case (a is array):
            roots : ndarray, same shape as a
                Root locations (NaN for invalid brackets)
            iterations : ndarray, same shape as a
                Iteration counts per problem
            converged : ndarray, same shape as a
                Convergence flags per problem
        
        Warnings
        --------
        Returns (np.nan, 0, False) for brackets without sign change.
        Each bracket must satisfy f(a) * f(b) < 0.
        
        See Also
        --------
        BrentSolver.find_root : Faster with same robustness
        _bisection_scalar : Scalar implementation details
        _bisection_vectorised : Vectorised implementation details
        
        Examples
        --------
        See class docstring for comprehensive examples.
        """

        return self._dispatch_root_bracket_method(
            func=func,
            a=a,
            b=b,
            func_params=func_params,
            scalar_bracket_method_func=_bisection_scalar,
            vector_bracket_method_func=_bisection_vectorised,
            tol=tol,
            max_iter=max_iter,
        )


class BrentSolver(Solver):
    """
    Brent's method solver for root finding.
    
    Implements Brent's algorithm, combining the robustness of bisection with
    the speed of inverse quadratic interpolation and secant methods. This is
    the RECOMMENDED general-purpose bracketing solver - faster than bisection
    with the same convergence guarantee.
    
    The solver adaptively chooses the best strategy at each iteration and
    automatically dispatches between scalar and vectorised execution.
    
    Attributes
    ----------
    name : SolverName
        Set to SolverName.BRENT
    method_type : MethodType
        Set to MethodType.HYBRID (uses both bracketing and interpolation)
    
    Examples
    --------
    Scalar root finding:
    
    >>> from numba import njit
    >>> 
    >>> @njit
    ... def func(x):
    ...     return x**2 - 4
    >>> 
    >>> solver = BrentSolver()
    >>> root, iters, conv = solver.find_root(
    ...     func, a=0.0, b=5.0, tol=1e-6, max_iter=100
    ... )
    >>> print(f"Root: {root:.6f}, Iterations: {iters}")
    Root: 2.000000, Iterations: 6
    >>> 
    >>> # Compare: Bisection needs ~23 iterations for same problem
    
    Vectorised root finding:
    
    >>> import numpy as np
    >>> a = np.array([0.0, -5.0, 1.0])
    >>> b = np.array([5.0, 0.0, 3.0])
    >>> roots, iters, conv = solver.find_root(
    ...     func, a=a, b=b, tol=1e-6
    ... )
    >>> print("Roots:", roots)
    Roots: [2.0, -2.0, 2.0]
    >>> print("Iterations:", iters)
    Iterations: [6, 6, 5]
    
    Transcendental function (where Brent excels):
    
    >>> @njit
    ... def transcendental(x):
    ...     return np.exp(x) - 2
    >>> 
    >>> root, iters, conv = solver.find_root(
    ...     transcendental, a=0.0, b=2.0, tol=1e-10
    ... )
    >>> print(f"Root: {root:.10f} (Expected: {np.log(2):.10f})")
    Root: 0.6931471806 (Expected: 0.6931471806)
    >>> print(f"Iterations: {iters}")
    Iterations: 8
    >>> # Bisection would need ~34 iterations for tol=1e-10
    
    See Also
    --------
    BisectionSolver : Simpler but slower (3-4x)
    NewtonRaphsonSolver : Faster but needs derivative and good guess
    _brent_scalar : Underlying scalar implementation
    _brent_vectorised : Underlying vectorised implementation
    """
    def __init__(self):
        self.name = SolverName.BRENT
        self.method_type = MethodType.HYBRID

    def find_root(
        self,
        func: Callable[[float], float],
        a: Union[npt.ArrayLike, float],
        b: Union[npt.ArrayLike, float],
        func_params: Optional[Union[Tuple[float, ...], npt.ArrayLike]] = None,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Find roots using Brent's method.
        
        Adaptively combines inverse quadratic interpolation, secant method,
        and bisection for fast, robust root finding.
        
        Automatically dispatches to scalar or vectorised implementation.
        Supports multi-dimensional arrays with shape preservation.
        
        Parameters
        ----------
        func : callable
            Function for which to find roots. Must be JIT-compiled (@njit)
            and have signature: func(x, *params) -> float.
        a : float or array_like
            Lower bracket bound(s). Must satisfy f(a) * f(b) < 0.
        b : float or array_like
            Upper bracket bound(s). Must have same shape as `a` and
            satisfy f(a) * f(b) < 0.
        func_params : tuple, array_like, or None, optional
            Additional parameters for func.
        tol : float, default=1e-6
            Convergence tolerance. Stops when |f(b)| < tol or
            |b-a| < tol where b is best current estimate.
        max_iter : int, default=100
            Maximum iterations allowed per problem. Brent typically
            converges in 5-15 iterations (much faster than bisection).
        
        Returns
        -------
        scalar case (a is scalar):
            root : float
                Root location (NaN if invalid bracket)
            iterations : int
                Number of iterations [0, max_iter]
            converged : bool
                True if converged, False if invalid bracket
        
        vectorised case (a is array):
            roots : ndarray, same shape as a
                Root locations (NaN for invalid brackets)
            iterations : ndarray, same shape as a
                Iteration counts per problem
            converged : ndarray, same shape as a
                Convergence flags per problem
        
        Warnings
        --------
        Returns (np.nan, 0, False) for brackets without sign change.
        Each bracket must satisfy f(a) * f(b) < 0.
        
        See Also
        --------
        BisectionSolver.find_root : Simpler but slower alternative
        _brent_scalar : Scalar implementation details
        _brent_vectorised : Vectorised implementation details
        
        Examples
        --------
        See class docstring for comprehensive examples.
        """
        return self._dispatch_root_bracket_method(
            func=func,
            a=a,
            b=b,
            func_params=func_params,
            scalar_bracket_method_func=_brent_scalar,
            vector_bracket_method_func=_brent_vectorised,
            tol=tol,
            max_iter=max_iter,
        )
