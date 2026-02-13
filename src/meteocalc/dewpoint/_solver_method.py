"""
Docstring for meteocalc.dewpoint._solver_method
"""

from rapid_roots.solvers import RootSolvers

from typing import Union, Callable
from meteocalc.vapor.core import Vapor
from meteocalc.vapor._vapor_equations import VaporEquation
from meteocalc.vapor._enums import EquationName, SurfaceType
from meteocalc.shared._enum_tools import parse_enum
from numba import njit
from meteocalc.dewpoint._dewpoint_equations import MagnusDewpointEquation

import numpy.typing as npt
import numpy as np






def get_dewpoint_using_solver(temp_k: Union[float, npt.ArrayLike], rh: Union[npt.ArrayLike],
                              surface_type: Union[str, SurfaceType], vapor_equation: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
    """
    Docstring for get_dewpoint_using_solver

    :param temp_k: Description
    :type temp_k: Union[float, npt.ArrayLike]
    :param rh: Description
    :type rh: Union[npt.ArrayLike]
    :param vapor_equation: Description
    :type vapor_equation: Union[str, EquationName]
    :param surface_type: Description
    :type surface_type: Union[str, SurfaceType]
    :return: Description
    :rtype: float | NDArray
    """
    n = len(temp_k)

    e_sat_air = Vapor.get_vapor_saturation(temp_k=temp_k, phase=surface_type, equation=vapor_equation)
    vapor_equation = Vapor.get_equation(equation=vapor_equation, phase=surface_type)
    
    e_actual = rh * e_sat_air

    a = temp_k - 100
    b = temp_k

    func_params = np.empty((n, 1), dtype=np.float64)
    func_params[:, 0] = e_actual.copy()

    dewpoint_objective_func = get_dewpoint_objective_function(vapor_equation=vapor_equation)

    roots, iters, converged = RootSolvers.get_root(
        func=dewpoint_objective_func,
        a=a,
        b=b,
        func_params=func_params,
        main_solver='brent',
        use_backup=True,
        backup_solvers=['bisection']
    )

    return roots, iters, converged

def get_dewpoint_objective_function(vapor_equation: VaporEquation):

    vapor_scalar_func = vapor_equation.get_jit_scalar_func()
    surface_constants = vapor_equation.get_constants()
    tuple_surface_constants = tuple(surface_constants)

    @njit
    def dewpoint_objective(x: float, e_actual: float, *surface_constants):
        """
        Docstring for dewpoint_objective
        
        :param x: Description
        :param e_sat: Description
        """
        e_sat = vapor_scalar_func(x, *tuple_surface_constants)

        return e_sat - e_actual

    return dewpoint_objective


if __name__ == "__main__":
    import time
    
    print("="*70)
    print("Testing Exact Dew Point Solver with RapidRoots")
    print("="*70)
    
    # ========================================================================
    # Test 1: Single Scalar Calculation
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: Single Scalar Calculation")
    print("="*70)
    
    temp_k = 293.15  # 20°C
    rh = 0.6         # 60%
    surface = 'water'
    equation = 'goff_gratch'
    
    print(f"\nInput:")
    print(f"  Temperature: {temp_k - 273.15:.2f}°C ({temp_k:.2f}K)")
    print(f"  Relative Humidity: {rh*100:.0f}%")
    print(f"  Surface: {surface}")
    print(f"  Equation: {equation}")
    
    start = time.perf_counter()
    roots, iters, converged = get_dewpoint_using_solver(
        temp_k=np.array([temp_k]),
        rh=np.array([rh]),
        surface_type=surface,
        vapor_equation=equation
    )
    elapsed = time.perf_counter() - start
    
    print(f"\nResults:")
    print(f"  Dew Point: {roots[0] - 273.15:.4f}°C ({roots[0]:.4f}K)")
    print(f"  Iterations: {iters[0]}")
    print(f"  Converged: {converged[0]}")
    print(f"  Time: {elapsed*1000:.3f} ms")
    
    # Expected: ~12°C for 20°C at 60% RH
    expected_td = 12.0
    error = abs(roots[0] - 273.15 - expected_td)
    print(f"\n  Expected: ~{expected_td:.1f}°C")
    print(f"  Error: {error:.4f}°C")
    
    if converged[0] and error < 0.5:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    # ========================================================================
    # Test 2: Vector Calculation (Small Array)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: Vector Calculation (10 points)")
    print("="*70)
    
    temps = np.linspace(273.15, 313.15, 10)  # -0°C to 40°C
    rhs = np.full(10, 0.6)
    
    print(f"\nInput:")
    print(f"  Temperatures: {temps[0]-273.15:.1f}°C to {temps[-1]-273.15:.1f}°C (10 points)")
    print(f"  RH: 60% (constant)")
    print(f"  Surface: water")
    
    start = time.perf_counter()
    roots, iters, converged = get_dewpoint_using_solver(
        temp_k=temps,
        rh=rhs,
        surface_type='water',
        vapor_equation='buck'
    )
    elapsed = time.perf_counter() - start
    
    print(f"\nResults:")
    print(f"  All converged: {np.all(converged)}")
    print(f"  Average iterations: {np.mean(iters):.1f}")
    print(f"  Max iterations: {np.max(iters)}")
    print(f"  Time: {elapsed*1000:.3f} ms")
    print(f"  Speed: {len(temps)/elapsed:.0f} calc/sec")
    
    print(f"\n  Sample dew points:")
    for i in [0, 4, 9]:
        print(f"    T={temps[i]-273.15:6.1f}°C, RH=60% → Td={roots[i]-273.15:6.2f}°C ({iters[i]} iter)")
    
    # Check monotonicity
    diffs = np.diff(roots)
    monotonic = np.all(diffs > 0)
    print(f"\n  Monotonic (increasing): {monotonic}")
    
    if np.all(converged) and monotonic:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    # ========================================================================
    # Test 3: Large Vector (Performance Test)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: Large Vector Performance (1000 points)")
    print("="*70)
    
    temps_large = np.linspace(273.15, 313.15, 1000)
    rhs_large = np.linspace(0.3, 0.9, 1000)
    
    print(f"\nInput:")
    print(f"  Points: 1000")
    print(f"  Temps: {temps_large[0]-273.15:.1f}°C to {temps_large[-1]-273.15:.1f}°C")
    print(f"  RH: {rhs_large[0]*100:.0f}% to {rhs_large[-1]*100:.0f}%")
    
    start = time.perf_counter()
    roots_large, iters_large, converged_large = get_dewpoint_using_solver(
        temp_k=temps_large,
        rh=rhs_large,
        surface_type='water',
        vapor_equation='goff_gratch'
    )
    elapsed = time.perf_counter() - start
    
    print(f"\nResults:")
    print(f"  All converged: {np.all(converged_large)} ({np.sum(converged_large)}/{len(converged_large)})")
    print(f"  Average iterations: {np.mean(iters_large):.2f}")
    print(f"  Max iterations: {np.max(iters_large)}")
    print(f"  Min iterations: {np.min(iters_large)}")
    print(f"  Time: {elapsed*1000:.2f} ms")
    print(f"  Speed: {len(temps_large)/elapsed:,.0f} calc/sec")
    print(f"  Time per point: {elapsed/len(temps_large)*1e6:.2f} µs")
    
    if np.all(converged_large):
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    # ========================================================================
    # Test 4: Ice vs Water Comparison
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 4: Ice vs Water (Frost Point > Dew Point)")
    print("="*70)
    
    temp_cold = np.array([263.15])  # -10°C
    rh_test = np.array([0.7])       # 70%
    
    print(f"\nInput:")
    print(f"  Temperature: {temp_cold[0]-273.15:.1f}°C")
    print(f"  RH: {rh_test[0]*100:.0f}%")
    
    # Dew point (water)
    roots_water, _, conv_water = get_dewpoint_using_solver(
        temp_k=temp_cold,
        rh=rh_test,
        surface_type='water',
        vapor_equation='buck'
    )
    
    # Frost point (ice)
    roots_ice, _, conv_ice = get_dewpoint_using_solver(
        temp_k=temp_cold,
        rh=rh_test,
        surface_type='ice',
        vapor_equation='buck'
    )
    
    td_water = roots_water[0] - 273.15
    tf_ice = roots_ice[0] - 273.15
    diff = tf_ice - td_water
    
    print(f"\nResults:")
    print(f"  Dew point (water):  {td_water:6.2f}°C (converged: {conv_water[0]})")
    print(f"  Frost point (ice):  {tf_ice:6.2f}°C (converged: {conv_ice[0]})")
    print(f"  Difference (Tf-Td): {diff:6.2f}°C")
    
    # Physics check: frost point should be higher than dew point below 0°C
    physics_correct = tf_ice > td_water and 0.5 < diff < 3.0
    
    print(f"\n  Frost point > Dew point: {tf_ice > td_water}")
    print(f"  Difference in expected range (0.5-3.0°C): {0.5 < diff < 3.0}")
    
    if physics_correct:
        print("  ✓ PASSED - Physics correct!")
    else:
        print("  ✗ FAILED - Physics violation!")
    
    # ========================================================================
    # Test 5: Different Vapor Equations Comparison
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 5: Comparing Different Vapor Equations")
    print("="*70)
    
    temp_test = np.array([293.15])
    rh_test = np.array([0.6])
    
    equations = ['buck', 'goff_gratch', 'hyland_wexler']
    
    print(f"\nInput: T={temp_test[0]-273.15:.1f}°C, RH={rh_test[0]*100:.0f}%")
    print(f"\nComparing equations:")
    
    results = {}
    for eq in equations:
        try:
            roots, iters, conv = get_dewpoint_using_solver(
                temp_k=temp_test,
                rh=rh_test,
                surface_type='water',
                vapor_equation=eq
            )
            results[eq] = {
                'td': roots[0] - 273.15,
                'iters': iters[0],
                'converged': conv[0]
            }
            print(f"  {eq:15s}: {results[eq]['td']:7.4f}°C ({results[eq]['iters']} iter, conv={results[eq]['converged']})")
        except Exception as e:
            print(f"  {eq:15s}: ERROR - {e}")
    
    # Check that all equations give similar results (within 0.1°C)
    if len(results) >= 2:
        tds = [r['td'] for r in results.values()]
        max_diff = np.max(tds) - np.min(tds)
        print(f"\n  Max difference between equations: {max_diff:.4f}°C")
        
        if max_diff < 0.1 and all(r['converged'] for r in results.values()):
            print("  ✓ PASSED - All equations agree!")
        else:
            print("  ✗ FAILED - Equations disagree or convergence issue")
    
    # ========================================================================
    # Test 6: Edge Cases
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 6: Edge Cases")
    print("="*70)
    
    edge_cases = [
        ("Very low RH", 293.15, 0.1, 'water'),
        ("Very high RH", 293.15, 0.99, 'water'),
        ("Cold temp", 243.15, 0.5, 'water'),
        ("Hot temp", 323.15, 0.5, 'water'),
    ]
    
    print(f"\nTesting edge cases:")
    all_passed = True
    
    for name, temp, rh, surface in edge_cases:
        try:
            roots, iters, conv = get_dewpoint_using_solver(
                temp_k=np.array([temp]),
                rh=np.array([rh]),
                surface_type=surface,
                vapor_equation='buck'
            )
            td = roots[0] - 273.15
            status = "✓" if conv[0] else "✗"
            print(f"  {status} {name:20s}: T={temp-273.15:6.1f}°C, RH={rh*100:4.0f}% → Td={td:7.2f}°C ({iters[0]} iter)")
            
            if not conv[0]:
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name:20s}: ERROR - {e}")
            all_passed = False
    
    if all_passed:
        print("\n  ✓ All edge cases PASSED")
    else:
        print("\n  ✗ Some edge cases FAILED")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("\nAll tests completed! Check results above.")
    print("\nKey metrics from large vector test:")
    print(f"  Speed: {len(temps_large)/elapsed:,.0f} calculations/second")
    print(f"  Average iterations: {np.mean(iters_large):.1f}")
    print(f"  Convergence rate: {np.sum(converged_large)/len(converged_large)*100:.1f}%")
    print("="*70)
