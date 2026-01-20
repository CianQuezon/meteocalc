"""
Core interface for Solvers

Author: Cian Quezon
"""

from typing import Callable, List, Optional, Tuple, Union

import warnings
import numpy as np
import numpy.typing as npt

from meteorological_equations.math.solvers._back_up_logic import _use_back_up_solvers, _update_converged_results
from meteorological_equations.math.solvers._enums import MethodType, SolverName
from meteorological_equations.math.solvers._solvers import (
    BisectionSolver,
    BrentSolver,
    NewtonRaphsonSolver,
    Solver,
)
from meteorological_equations.shared._enum_tools import parse_enum
from meteorological_equations.math.solvers._types_and_maps import SolverMap

class RootSolvers:
    
    @staticmethod
    def list_root_solvers() -> List[str]:
        """
        List all available root-finding solvers.

        Returns a list of solver names that can be used with the `get_root` method.
        Each solver implements a different numerical algorithm for finding roots of
        equations, with varying characteristics in terms of speed, robustness, and
        input requirements.

        Returns
        -------
        list of str
            Names of all available solvers. Each name can be passed as the
            `main_solver` or included in `backup_solvers` parameter when calling
            `get_root`. Solver names are:
            
            - 'newton' : Newton-Raphson method (fast, requires derivative)
            - 'brent' : Brent's method (robust hybrid, recommended default)
            - 'bisection' : Bisection method (reliable, slower)

        Notes
        -----
        Solver Characteristics:

        **Newton-Raphson ('newton')**
            - Type: Open method
            - Requires: Initial guess (x0), derivative (func_prime)
            - Speed: Very fast (quadratic convergence)
            - Robustness: Can fail if initial guess is poor or derivative is zero
            - Best for: When you have a good initial guess and can provide derivative

        **Brent's Method ('brent')**
            - Type: Hybrid (can use open or bracket mode)
            - Requires: Bracket [a, b] or initial guess (x0), or both
            - Speed: Fast (superlinear convergence)
            - Robustness: Very robust, combines bisection, secant, and inverse quadratic
            - Best for: General purpose use (recommended as primary solver)

        **Bisection ('bisection')**
            - Type: Bracket method
            - Requires: Bracket [a, b] with f(a)*f(b) < 0
            - Speed: Slower (linear convergence)
            - Robustness: Always converges if bracket is valid
            - Best for: Guaranteed convergence when you have a valid bracket

        Default Backup Chain:
            If not specified, the default backup solver chain is ['brent', 'bisection'],
            which provides excellent robustness for most applications.

        Examples
        --------
        List all available solvers:

        >>> from meteorological_equations.math.solvers import RootSolvers
        >>> 
        >>> solvers = RootSolvers.list_root_solvers()
        >>> print(solvers)
        ['newton', 'brent', 'bisection']

        Use different solvers:

        >>> import numpy as np
        >>> from numba import njit
        >>> 
        >>> @njit
        ... def f(x):
        ...     return x**3 - 8
        >>> 
        >>> @njit
        ... def f_prime(x):
        ...     return 3 * x**2
        >>> 
        >>> # Use Newton-Raphson
        >>> root, iters, conv = RootSolvers.get_root(
        ...     func=f,
        ...     x0=2.5,
        ...     func_prime=f_prime,
        ...     main_solver='newton'
        ... )
        >>> print(f"Newton: {root:.6f} in {iters} iterations")
        Newton: 2.000000 in 4 iterations
        >>> 
        >>> # Use Brent's method
        >>> root, iters, conv = RootSolvers.get_root(
        ...     func=f,
        ...     a=0.0,
        ...     b=5.0,
        ...     main_solver='brent'
        ... )
        >>> print(f"Brent: {root:.6f} in {iters} iterations")
        Brent: 2.000000 in 6 iterations
        >>> 
        >>> # Use Bisection
        >>> root, iters, conv = RootSolvers.get_root(
        ...     func=f,
        ...     a=0.0,
        ...     b=5.0,
        ...     main_solver='bisection'
        ... )
        >>> print(f"Bisection: {root:.6f} in {iters} iterations")
        Bisection: 2.000000 in 26 iterations

        Create custom backup chain:

        >>> # Get available solvers
        >>> available = RootSolvers.list_root_solvers()
        >>> 
        >>> # Create custom chain: try Newton first, then reliable methods
        >>> custom_chain = ['brent', 'bisection']
        >>> 
        >>> # Verify all are valid
        >>> assert all(solver in available for solver in custom_chain)
        >>> 
        >>> # Use custom chain
        >>> root, iters, conv = RootSolvers.get_root(
        ...     func=f,
        ...     x0=2.5,
        ...     a=0.0,
        ...     b=5.0,
        ...     func_prime=f_prime,
        ...     main_solver='newton',
        ...     backup_solvers=custom_chain
        ... )

        Dynamically check if solver is available:

        >>> def use_solver_if_available(solver_name):
        ...     '''Use solver only if it exists in current version.'''
        ...     available = RootSolvers.list_root_solvers()
        ...     
        ...     if solver_name in available:
        ...         print(f"Using {solver_name}")
        ...         return solver_name
        ...     else:
        ...         print(f"{solver_name} not available, using brent")
        ...         return 'brent'
        >>> 
        >>> solver = use_solver_if_available('newton')
        Using newton
        >>> 
        >>> solver = use_solver_if_available('nonexistent')
        nonexistent not available, using brent

        Performance comparison of all solvers:

        >>> import time
        >>> 
        >>> @njit
        ... def complex_func(x):
        ...     return np.sin(x) * np.exp(-x**2) - 0.1
        >>> 
        >>> @njit
        ... def complex_func_prime(x):
        ...     return np.cos(x) * np.exp(-x**2) - 2*x*np.sin(x)*np.exp(-x**2)
        >>> 
        >>> # Test all available solvers with brackets
        >>> for solver in ['brent', 'bisection']:
        ...     start = time.time()
        ...     root, iters, conv = RootSolvers.get_root(
        ...         func=complex_func,
        ...         a=-2.0,
        ...         b=2.0,
        ...         main_solver=solver,
        ...         use_backup=False
        ...     )
        ...     elapsed = time.time() - start
        ...     print(f"{solver:12s}: {iters:3d} iters, {elapsed*1000:.2f}ms")
        brent       :   8 iters, 0.45ms
        bisection   :  26 iters, 0.62ms
        >>> 
        >>> # Test Newton with initial guess
        >>> start = time.time()
        >>> root, iters, conv = RootSolvers.get_root(
        ...     func=complex_func,
        ...     x0=1.0,
        ...     func_prime=complex_func_prime,
        ...     main_solver='newton',
        ...     use_backup=False
        ... )
        >>> elapsed = time.time() - start
        >>> print(f"{'newton':12s}: {iters:3d} iters, {elapsed*1000:.2f}ms")
        newton      :   4 iters, 0.28ms

        See Also
        --------
        get_root : Main function for finding roots using these solvers
        SolverName : Enum containing solver name constants

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
        max_iter: int = 100,
        main_solver: Union[SolverName, str] = SolverName.NEWTON,
        use_backup: bool = True,
        backup_solvers: Optional[List[Union[str, SolverName]]] = None,
    ) -> Union[
        Tuple[float, int, bool],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]],
    ]:
        """
        Find roots of a function using numerical solvers with automatic backup.
        
        This function provides a unified interface for root-finding that automatically
        falls back to backup solvers if the primary solver fails to converge. It
        supports both scalar and vectorized inputs, and works with open, bracket,
        and hybrid methods.
        
        Parameters
        ----------
        func : callable
            Function for which to find roots. Should accept a float and return a float.
            For parametric functions, use `func_params` to pass additional arguments.
            
        a : float or array_like, optional
            Lower bracket bound(s). Required for bracket-based methods (BISECTION)
            and can be used as fallback for hybrid methods (BRENT). If array, must
            match the length of other array inputs.
            
        b : float or array_like, optional
            Upper bracket bound(s). Required for bracket-based methods. For each
            bracket pair, must satisfy func(a) * func(b) < 0.
            
        x0 : float or array_like, optional
            Initial guess(es) for the root. Required for open methods (NEWTON) and
            can be used as primary attempt for hybrid methods.
            
        func_prime : callable, optional
            Derivative of `func`. Required for Newton-Raphson method. Should accept
            a float and return a float.
            
        func_params : array_like, optional
            Additional parameters to pass to `func` and `func_prime`. Can be:
            - None: No additional parameters
            - 1D array: Shared parameters for all elements
            - 2D array: Different parameters per element (for vectorized calls)
            
        tol : float, default=1e-6
            Convergence tolerance. Solver stops when |func(x)| < tol or when the
            change in x between iterations falls below tol.
            
        max_iter : int, default=100
            Maximum number of iterations allowed per solver attempt.
            
        main_solver : SolverName or str, default='newton'
            Primary solver to use. Available options:
            - 'newton' : Newton-Raphson (open method, requires x0 and func_prime)
            - 'brent'  : Brent's method (hybrid, can use x0 or [a,b] or both)
            - 'bisection' : Bisection (bracket method, requires a and b)
            
        use_backup : bool, default=True
            Whether to use backup solvers for unconverged elements. If False,
            returns results from primary solver even if some elements didn't converge.
            
        backup_solvers : list of SolverName or str, optional
            Ordered list of backup solvers to try on unconverged elements.
            Default is ['brent', 'bisection']. Solvers are tried in order until
            all elements converge or all solvers are exhausted.
        
        Returns
        -------
        roots : float or ndarray of float
            Root value(s). NaN for unconverged elements.
            
        iterations : int or ndarray of int
            Number of iterations used for each element.
            
        converged : bool or ndarray of bool
            Convergence status. True if converged within tolerance, False otherwise.
        
        Raises
        ------
        ValueError
            If required inputs are missing for the specified solver type, or if
            no inputs are provided to determine the problem size.
        
        Notes
        -----
        Solver Requirements:
        - Open methods (NEWTON): Require `x0`. Newton also requires `func_prime`.
        - Bracket methods (BISECTION): Require both `a` and `b`.
        - Hybrid methods (BRENT): Can use `x0` for open mode, or `a` and `b` for
        bracket mode, or both (tries open first, then bracket for failures).
        
        Backup Solver Chain:
        When `use_backup=True`, the function automatically tries backup solvers
        on elements that failed to converge with the primary solver. This ensures
        maximum reliability without requiring manual intervention.
        
        Vectorization:
        All inputs can be either scalar (for single root-finding) or array-like
        (for finding multiple roots simultaneously). Array inputs must have
        compatible shapes.
        
        Examples
        --------
        Find a single root using Brent's method:
        
        >>> def f(x):
        ...     return x**2 - 4
        >>> 
        >>> root, iters, converged = RootSolvers.get_root(
        ...     func=f,
        ...     a=0.0,
        ...     b=5.0,
        ...     main_solver='brent'
        ... )
        >>> print(f"Root: {root:.6f}, Converged: {converged}")
        Root: 2.000000, Converged: True
        
        Find multiple roots with Newton-Raphson and automatic backup:
        
        >>> import numpy as np
        >>> 
        >>> def f(x):
        ...     return x**3 - 8
        >>> 
        >>> def fp(x):
        ...     return 3*x**2
        >>> 
        >>> x0 = np.array([1.0, 1.5, 2.0, 2.5])
        >>> 
        >>> roots, iters, conv = RootSolvers.get_root(
        ...     func=f,
        ...     x0=x0,
        ...     func_prime=fp,
        ...     main_solver='newton',
        ...     use_backup=True,
        ...     backup_solvers=['brent', 'bisection']
        ... )
        >>> print(f"All converged: {np.all(conv)}")
        All converged: True
        >>> print(f"Roots: {roots}")
        Roots: [2. 2. 2. 2.]
        
        Use hybrid solver with both initial guess and bracket:
        
        >>> roots, iters, conv = RootSolvers.get_root(
        ...     func=f,
        ...     x0=1.0,      # Try open method first
        ...     a=0.0,       # Fall back to bracket if needed
        ...     b=3.0,
        ...     main_solver='brent'
        ... )
        
        Parametric function with different parameters per element:
        
        >>> @njit
        ... def parametric_func(x, a, b):
        ...     return a * x**2 + b
        >>> 
        >>> # Solve: a*x^2 + b = 0 for different (a, b) pairs
        >>> func_params = np.array([
        ...     [1.0, -4.0],   # x^2 - 4 = 0, root at x=2
        ...     [2.0, -8.0],   # 2x^2 - 8 = 0, root at x=2
        ...     [0.5, -2.0],   # 0.5x^2 - 2 = 0, root at x=2
        ... ])
        >>> 
        >>> roots, iters, conv = RootSolvers.get_root(
        ...     func=parametric_func,
        ...     a=np.array([0.0, 0.0, 0.0]),
        ...     b=np.array([5.0, 5.0, 5.0]),
        ...     func_params=func_params,
        ...     main_solver='brent'
        ... )
        >>> print(f"All roots: {roots}")
        All roots: [2. 2. 2.]
        
        See Also
        --------
        list_root_solvers : List all available solvers
        """
        results = None

        if backup_solvers is None:
            backup_solvers = [SolverName.BRENT, SolverName.BISECTION]

        n_problems, _ = RootSolvers._get_problem_size(x0, a=a, b=b)

        func_params = RootSolvers._standardise_func_params(func_params=func_params, n_problems=n_problems)

        solver_enum = parse_enum(main_solver, SolverName)
        solver = SolverMap[solver_enum]()

        method_type = solver.get_method_type()
        
        if method_type == MethodType.HYBRID:

            if x0 is not None:
                try:
                    results = solver.find_root(
                        func=func,
                        func_prime=func_prime,
                        x0=x0,
                        func_params=func_params,
                        tol=tol,
                        max_iter=max_iter
                    )
                    
                    _, _, converged_flags = results

                    if np.all(converged_flags):
                        return results
            
                except Exception as e:
                    warnings.warn(
                        f"Open interface for hybrid solver {solver_enum.value} failed: {e}",
                        stacklevel=2,
                    )

            if a is not None and b is not None:

                try:
                    bracket_results = solver.find_root(
                        func=func,
                        a=a,
                        b=b,
                        func_params=func_params,
                        tol=tol,
                        max_iter=max_iter,
                    )
                   
                    if results is None:
                        results = bracket_results
                    
                    else:
                        roots, iterations, converged_flags = results
                        bracket_roots, bracket_iterations, bracket_converged_flags = bracket_results

                        open_roots_arr = np.asarray(roots, dtype=np.float64)
                        
                        if open_roots_arr.ndim == 0:

                            if not converged_flags:
                                results = bracket_results
                        
                        else:
                            unconverged_idx = np.where(np.logical_not(converged_flags))[0]

                            if len(unconverged_idx) > 0:
                                _update_converged_results(
                                    roots=roots,
                                    iterations=iterations,
                                    converged_flag=converged_flags,
                                    unconverged_idx=unconverged_idx,
                                    updated_roots=bracket_roots[unconverged_idx],
                                    updated_iterations=bracket_iterations[unconverged_idx],
                                    updated_converged_flag=bracket_converged_flags[unconverged_idx]
                                )

                                results = (roots, iterations, converged_flags)

                except Exception as e:

                    warnings.warn(
                        f"Bracket interface for hybrid solver {solver_enum.value} failed: {e}",
                        stacklevel=2,
                    )
                    
        elif method_type == MethodType.OPEN:

            try:
                results = solver.find_root(
                    func=func,
                    func_prime=func_prime,
                    x0=x0,
                    func_params=func_params,
                    tol=tol,
                    max_iter=max_iter,
                )

            except Exception as e:
                warnings.warn(
                    f"Open solver {solver_enum.value} failed: {e}",
                    stacklevel=2,
                )

        elif method_type == MethodType.BRACKET:
            
            try:
                results = solver.find_root(
                    func=func, 
                    a=a, 
                    b=b, 
                    func_params=func_params, 
                    tol=tol, 
                    max_iter=max_iter
                )
            
            except Exception as e:
                warnings.warn(
                    f"Bracket solver {solver_enum.value} failed: {e}",
                    stacklevel=2,
                )
        
        
        if results is None:
            warnings.warn(
                f"Primary solver {solver_enum.value} failed completely. "
                f"{'Creating substitute results for backup chain.' if use_backup else 'Returning unconverged results.'}",
                stacklevel=2
            )
        
            results = RootSolvers._create_substitute_results(
                x0=x0,
                a=a,
                b=b,
                max_iter=max_iter
            )
            
        _, _, converged_flags = results

        if np.all(converged_flags):
            return results
        
        if not use_backup:
            return results
        

        return _use_back_up_solvers(
            func=func, results=results,
            a=a, b=b, x0=x0, tol=tol, max_iter=max_iter,
            func_prime=func_prime, func_params=func_params,
            backup_solvers=backup_solvers
        )
        
    @staticmethod
    def _create_substitute_results(
        x0: Optional[Union[float, npt.ArrayLike]] = None,
        a: Optional[Union[float, npt.ArrayLike]] = None,
        b: Optional[Union[float, npt.ArrayLike]] = None,
        max_iter: int = 100
    ) -> Tuple[Union[float, npt.NDArray], Union[int, npt.NDArray], Union[bool, npt.NDArray]]:
        """

            Create substitute unconverged results when primary solver fails completely.

            Generates result tuples filled with "failure" values (NaN roots, maximum
            iterations, False convergence flags) matching the shape of provided inputs.
            Used to initialize results when the primary solver fails to produce any output,
            allowing the backup solver chain to attempt the problem from scratch.

            The function infers the result shape from the first non-None input parameter
            in priority order: x0, then a, then b. This ensures consistent handling
            regardless of which solver type (open, bracket, or hybrid) is being used.

            Parameters
            ----------
            x0 : float or array_like, optional
                Initial guess(es) for root-finding. If provided, used to determine
                the shape of output arrays. Takes priority over a and b for shape
                determination.
            a : float or array_like, optional
                Lower bracket bound(s). Used to determine output shape if x0 is None.
                Takes priority over b.
            b : float or array_like, optional
                Upper bracket bound(s). Used to determine output shape only if both
                x0 and a are None.
            max_iter : int, default=100
                Maximum iteration count to assign to the substitute results. This
                value indicates that no actual iterations were performed by the
                primary solver.

            Returns
            -------
            roots : float or ndarray of float64
                Substitute root values. Scalar NaN if inputs are scalar, or array
                of NaN values matching input length if inputs are arrays.
            iterations : int or ndarray of int64
                Substitute iteration counts. Scalar max_iter if inputs are scalar,
                or array of max_iter values if inputs are arrays.
            converged : bool or ndarray of bool
                Substitute convergence flags. Scalar False if inputs are scalar,
                or array of False values if inputs are arrays.

            Raises
            ------
            ValueError
                If all of x0, a, and b are None, making it impossible to determine
                the required output shape.

            Notes
            -----
            Shape Determination Priority:
                The function checks inputs in this order: x0 → a → b. This priority
                ensures that the most commonly used parameter for each solver type
                (open methods use x0, bracket methods use a/b) is checked first.

            Scalar vs Array:
                Automatically detects whether inputs represent a single problem (scalar
                or 0-D array) or multiple problems (1-D+ array) by checking ndim.

            Use in Backup Chain:
                When the primary solver returns None, this creates "all unconverged"
                results that backup solvers can process. All elements are marked as
                unconverged (converged=False), so the backup chain will attempt all.

            See Also
            --------
            get_root : Main function that uses this helper
            _use_back_up_solvers : Backup chain that processes these results
        
        """

        n_problems, is_scalar = RootSolvers._get_problem_size(x0=x0, a=a, b=b)
        
        if is_scalar:
            return (np.nan, max_iter,False)
        
        else:
            return(
                np.full(n_problems, np.nan, dtype=np.float64),
                np.full(n_problems, max_iter, dtype=np.int64),
                np.full(n_problems, False, dtype=bool)
            )
        
    @staticmethod
    def _get_problem_size(
        x0: Optional[Union[float, npt.ArrayLike]] = None,
        a: Optional[Union[float, npt.ArrayLike]] = None,
        b: Optional[Union[float, npt.ArrayLike]] = None,
    ) -> Tuple[int, bool]:
        """
            Determine the number of problems from input arrays.
            
            Checks x0, then a, then b (in priority order) to determine how many
            root-finding problems are being solved simultaneously.
            
            Parameters
            ----------
            x0 : float or array_like, optional
                Initial guess(es). Takes priority over a and b.
            a : float or array_like, optional
                Lower bracket bound(s). Used if x0 is None.
            b : float or array_like, optional
                Upper bracket bound(s). Used if x0 and a are None.
            
            Returns
            -------
            n_problems : int
                Number of problems. Returns 1 for scalar inputs, len(array) for
                array inputs.
            
            Raises
            ------
            ValueError
                If all inputs are None.
        """

        array_size = None
        
        if x0 is not None:
           array_size = np.asarray(x0)        
        elif a is not None:
            array_size = np.asarray(a)
        elif b is not None:
            array_size = np.asarray(b)
        
        if array_size is None:
            raise ValueError(
                "Cannot determine problem size. It requires at least one of the following: x0, a, or b. "
                "Must be provided to create a substitute results for backup solvers"
            )
        
        is_scalar = (array_size.ndim == 0)

        if is_scalar:
            return 1, True
        else:
            return len(array_size), False
        
    @staticmethod
    def _standardise_func_params(
        func_params: Optional[npt.ArrayLike],
        n_problems: int
    ) -> Optional[npt.NDArray]:
        """
        Standardize func_params to 2D array format.
        
        Parameters
        ----------
        func_params : array_like or None
            Function parameters to standardize.
        n_problems : int
            Number of problems (from _determine_problem_size).
        
        Returns
        -------
        func_params_2d : ndarray or None
            Standardized parameters as (n_problems, n_params) array.
        """

        if func_params is None:
            return None
        
        func_params = np.asarray(func_params)

        if func_params.ndim == 0:
            return np.full((n_problems, 1), func_params, dtype=np.float64)
        elif func_params.ndim == 1:
            return np.tile(func_params, (n_problems, 1))
        elif func_params.ndim == 2:
            if func_params.shape[0] != n_problems:
                raise ValueError(
                    f"func_params has {func_params.shape[0]} rows but expected "
                    f"{n_problems} rows"
                )
            return func_params
        else:
            raise ValueError(f"func_params must be 0D, 1D, or 2D, got {func_params.ndim}D")