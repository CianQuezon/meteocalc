"""
Core interface for Solvers

Author: Cian Quezon
"""


from meteorological_equations.math.solvers._enums import SolverName
from meteorological_equations.math.solvers._solvers import (
    BisectionSolver,
    BrentSolver,
    NewtonRaphsonSolver,
)

SolverMap = {
    SolverName.NEWTON: NewtonRaphsonSolver,
    SolverName.BISECTION: BisectionSolver,
    SolverName.BRENT: BrentSolver,
}
