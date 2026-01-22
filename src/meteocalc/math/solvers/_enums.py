"""
Enums for the solvers to get the root

Author: Cian Quezon
"""

from enum import Enum


class SolverName(Enum):
    """
    Enumeration of available root-finding solver algorithms.

    This enum identifies which numerical algorithm to use for root finding.
    Each solver has different characteristics, convergence properties, and
    input requirements.

    Attributes
    ----------
    NEWTON : str
        Newton-Raphson method. Fast quadratic convergence but requires
        derivative and good initial guess. Value: 'newton'
    BRENT : str
        Brent's method. Robust bracket-based method combining inverse
        quadratic interpolation, secant method, and bisection. Recommended
        for most bracket-based problems. Value: 'brent'
    BISECTION : str
        Bisection method. Simple and robust bracket-based method with
        guaranteed linear convergence. Slower but very reliable. Value: 'bisection'

    See Also
    --------
    MethodType : Categorizes solvers by input requirements
    RootSolvers.get_root : Main interface that uses these solver names
    """

    NEWTON = "newton"
    BRENT = "brent"
    BISECTION = "bisection"


class MethodType(Enum):
    """
    Enumeration of solver method categories based on input requirements.

    This enum classifies root-finding methods by their algorithmic approach
    and input requirements. Used internally to determine which inputs are
    needed and how to structure solver calls.

    Attributes
    ----------
    OPEN : str
        Open methods that start from an initial guess and iterate without
        requiring a bracket. May diverge if initial guess is poor.
        Requires: x0 (initial guess), optionally func_prime (derivative).
        Examples: Newton-Raphson, Secant method.
        Value: 'open'
    BRACKET : str
        Bracketing methods that require an interval [a, b] where the function
        changes sign. Guaranteed to converge within the bracket.
        Requires: a (lower bound), b (upper bound).
        Examples: Bisection, Brent, Ridders.
        Value: 'bracket'
    HYBRID : str
        Hybrid methods that can use both bracket and initial guess, combining
        advantages of open and bracketing approaches.
        Requires: a, b, and optionally x0, func_prime.
        Examples: Brent with initial guess, Newton with bracketing fallback.
        Value: 'hybrid'
    CUSTOM : str
        User-defined or custom methods that don't fit standard categories.
        Input requirements determined by the specific implementation.
        Value: 'custom'

    See Also
    --------
    SolverName : Identifies specific solver algorithms
    BaseSolver.get_method_type : Returns the MethodType for a solver
    RootSolvers.get_root : Uses MethodType for input validation
    """

    OPEN = "open"
    BRACKET = "bracket"
    HYBRID = "hybrid"
    CUSTOM = "custom"
