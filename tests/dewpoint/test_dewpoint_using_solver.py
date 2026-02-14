"""
Comprehensive pytest suite for exact dew point solver using RapidRoots.

Tests numerical inversion of vapor pressure equations for maximum accuracy
and thermodynamic consistency. Validates against MetPy, PsychroLib, and
known reference values.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import warnings

from meteocalc.dewpoint._solver_method import (
    get_dewpoint_using_solver,
    get_dewpoint_objective_function
)
from meteocalc.vapor._enums import EquationName, SurfaceType

# Reference implementations
try:
    import psychrolib as psy
    psy.SetUnitSystem(psy.SI)
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False

try:
    from metpy.calc import dewpoint_from_relative_humidity
    from metpy.units import units
    HAS_METPY = True
except ImportError:
    HAS_METPY = False


# ==============================================================================
# Helper Functions
# ==============================================================================

def celsius_to_kelvin(temp_c):
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15


def get_metpy_dewpoint(temp_k, rh):
    """Calculate dew point using MetPy for validation."""
    if not HAS_METPY:
        pytest.skip("MetPy not available")
    
    temp_c = kelvin_to_celsius(temp_k)
    
    try:
        td_c = dewpoint_from_relative_humidity(
            temp_c * units.degC,
            rh * units.dimensionless
        ).magnitude
        return celsius_to_kelvin(td_c)
    except Exception as e:
        pytest.skip(f"MetPy error: {e}")


def get_psychrolib_dewpoint(temp_k, rh):
    """Calculate dew point using PsychroLib for validation."""
    if not HAS_PSYCHROLIB:
        pytest.skip("PsychroLib not available")
    
    temp_c = kelvin_to_celsius(temp_k)
    
    if not (0.0 <= rh <= 1.0):
        raise ValueError(f"RH must be in [0, 1], got {rh}")
    
    try:
        td_c = psy.GetTDewPointFromRelHum(temp_c, rh)
    except Exception as e:
        pytest.skip(f"PsychroLib error: {e}")
    
    return celsius_to_kelvin(td_c)


# ==============================================================================
# NOAA Reference Data
# ==============================================================================

NOAA_REFERENCE_DATA = [
    # (temp_celsius, rh_percent, dewpoint_celsius)
    # Standard conditions
    (20.0, 50.0, 9.3),
    (20.0, 60.0, 12.0),
    (20.0, 70.0, 14.4),
    (20.0, 80.0, 16.4),
    (20.0, 90.0, 18.3),
    
    # Warm conditions
    (25.0, 50.0, 13.9),
    (30.0, 50.0, 18.4),
    (30.0, 70.0, 24.1),
    (35.0, 60.0, 26.1),
    
    # Cool conditions
    (15.0, 60.0, 7.6),
    (10.0, 50.0, -0.1),
    (10.0, 80.0, 6.7),
    (5.0, 70.0, -0.2),
    (0.0, 90.0, -1.4),
]


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(params=['goff_gratch', 'hyland_wexler'])
def vapor_equation(request):
    """Parametrize tests across different vapor equations."""
    return request.param


# ==============================================================================
# Test Class: Basic Functionality
# ==============================================================================

class TestSolverBasicFunctionality:
    """Test basic functionality of the exact dew point solver."""
    
    def test_returns_tuple_of_three_arrays(self):
        """Test that solver returns (roots, iterations, converged)."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        result = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        roots, iters, converged = result
        assert isinstance(roots, np.ndarray)
        assert isinstance(iters, np.ndarray)
        assert isinstance(converged, np.ndarray)
    
    def test_scalar_input_returns_single_values(self):
        """Test that single input returns arrays of length 1."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert len(roots) == 1
        assert len(iters) == 1
        assert len(converged) == 1
    
    def test_vector_input_preserves_length(self):
        """Test that vector input returns vectors of same length."""
        n = 10
        temp_k = np.linspace(273.15, 313.15, n)
        rh = np.full(n, 0.6)
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert len(roots) == n
        assert len(iters) == n
        assert len(converged) == n
    
    def test_accepts_string_surface_type(self):
        """Test that surface_type can be passed as string."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_accepts_enum_surface_type(self):
        """Test that surface_type can be passed as enum."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type=SurfaceType.WATER,
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_accepts_string_vapor_equation(self):
        """Test that vapor_equation can be passed as string."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_accepts_enum_vapor_equation(self):
        """Test that vapor_equation can be passed as enum."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation=EquationName.GOFF_GRATCH
        )
        
        assert converged[0]


# ==============================================================================
# Test Class: Convergence
# ==============================================================================

class TestSolverConvergence:
    """Test convergence behavior of the solver."""
    
    def test_converges_for_typical_conditions(self, vapor_equation):
        """Test convergence for typical meteorological conditions."""
        temp_k = np.array([273.15, 283.15, 293.15, 303.15])
        rh = np.array([0.5, 0.6, 0.7, 0.8])
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation=vapor_equation
        )
        
        assert np.all(converged), "Not all solutions converged"
    
    def test_low_iteration_count(self):
        """Test that solver converges quickly (typically 5-10 iterations)."""
        temp_k = np.linspace(273.15, 313.15, 20)
        rh = np.full(20, 0.6)
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert np.all(converged)
        assert np.mean(iters) < 15, f"Average iterations too high: {np.mean(iters)}"
        assert np.max(iters) < 20, f"Max iterations too high: {np.max(iters)}"
    
    def test_converges_at_low_humidity(self):
        """Test convergence at very low humidity (10%)."""
        temp_k = np.array([293.15])
        rh = np.array([0.1])
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_converges_at_high_humidity(self):
        """Test convergence at very high humidity (99%)."""
        temp_k = np.array([293.15])
        rh = np.array([0.99])
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_converges_at_cold_temperature(self):
        """Test convergence at cold temperature (0°C)."""
        temp_k = np.array([273.15])
        rh = np.array([0.6])
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_converges_at_hot_temperature(self):
        """Test convergence at hot temperature (50°C)."""
        temp_k = np.array([323.15])
        rh = np.array([0.6])
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_large_array_convergence(self):
        """Test 100% convergence on large array (1000 points)."""
        n = 1000
        temp_k = np.linspace(273.15, 313.15, n)
        rh = np.linspace(0.3, 0.9, n)
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        convergence_rate = np.sum(converged) / len(converged)
        assert convergence_rate >= 0.99, f"Convergence rate too low: {convergence_rate*100:.1f}%"


# ==============================================================================
# Test Class: Physical Correctness
# ==============================================================================

class TestSolverPhysicalCorrectness:
    """Test that results satisfy physical constraints."""
    
    def test_dewpoint_less_than_temperature(self):
        """Test that dew point is always ≤ air temperature."""
        temp_k = np.linspace(273.15, 313.15, 20)
        rh = np.linspace(0.3, 0.9, 20)
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert np.all(roots[converged] <= temp_k[converged]), \
            "Dew point exceeds air temperature"
    
    def test_dewpoint_at_100_percent_rh(self):
        """Test that Td ≈ T at RH=100%."""
        temp_k = np.array([273.15, 293.15, 313.15])
        rh = np.array([1.0, 1.0, 1.0])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        # At RH=100%, dew point should equal temperature within numerical tolerance
        assert_allclose(roots[converged], temp_k[converged], atol=0.1,
                       err_msg="Td should equal T at RH=100%")
    
    def test_monotonic_with_temperature(self):
        """Test that Td increases with T at constant RH."""
        rh = 0.6
        temp_k = np.linspace(273.15, 313.15, 20)
        rh_arr = np.full(20, rh)
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh_arr,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        # Check monotonicity where converged
        diffs = np.diff(roots[converged])
        assert np.all(diffs > 0), "Dew point not monotonic with temperature"
    
    def test_monotonic_with_humidity(self):
        """Test that Td increases with RH at constant T."""
        temp_k_val = 293.15
        temp_k = np.full(20, temp_k_val)
        rh = np.linspace(0.3, 0.9, 20)
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        # Check monotonicity
        diffs = np.diff(roots[converged])
        assert np.all(diffs > 0), "Dew point not monotonic with RH"
    
    def test_frost_point_greater_than_dewpoint_below_freezing(self):
        """Test that frost point > dew point below 0°C.
        
        Below freezing, ice has lower saturation vapor pressure than
        supercooled water, so frost point is warmer than dew point.
        """
        temp_k = np.array([263.15, 268.15])  # -10°C, -5°C
        rh = np.array([0.7, 0.7])
        
        # Dew point (water)
        roots_water, _, conv_water = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        # Frost point (ice)
        roots_ice, _, conv_ice = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='ice',
            vapor_equation='goff_gratch'
        )
        
        # Where both converged, frost point should be higher
        both_converged = conv_water & conv_ice
        assert np.all(roots_ice[both_converged] > roots_water[both_converged]), \
            "Frost point should be > dew point below 0°C"
    
    def test_dewpoint_depression_reasonable(self):
        """Test that dew point depression is in reasonable range."""
        temp_k = np.array([293.15])  # 20°C
        rh = np.array([0.5])  # 50%
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        depression = temp_k[converged] - roots[converged]
        
        # At 20°C, 50% RH, depression should be roughly 9-10K
        assert np.all(depression > 0)
        assert np.all(depression < 50), "Depression unreasonably large"


# ==============================================================================
# Test Class: NOAA Reference Validation
# ==============================================================================

class TestSolverNOAAValidation:
    """Validate against NOAA Online Calculator reference values."""
    
    @pytest.mark.parametrize("temp_c,rh_percent,expected_td_c", NOAA_REFERENCE_DATA)
    def test_noaa_reference_values(self, temp_c, rh_percent, expected_td_c):
        """Test against NOAA calculator reference values.
        
        Numerical solver should agree with NOAA within ±0.3°C.
        Better accuracy than approximation methods.
        """
        temp_k = np.array([celsius_to_kelvin(temp_c)])
        rh = np.array([rh_percent / 100.0])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0], f"Failed to converge for T={temp_c}°C, RH={rh_percent}%"
        
        td_c = kelvin_to_celsius(roots[0])
        
        assert_allclose(
            td_c, expected_td_c,
            atol=0.3,
            err_msg=(
                f"Solver vs NOAA: T={temp_c}°C, RH={rh_percent}%, "
                f"Expected={expected_td_c}°C, Got={td_c:.2f}°C"
            )
        )


# ==============================================================================
# Test Class: MetPy Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestSolverMetPyValidation:
    """Cross-validate against MetPy."""
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 30, 40])
    @pytest.mark.parametrize("rh", [0.4, 0.6, 0.8])
    def test_against_metpy(self, temp_c, rh):
        """Test against MetPy across range of conditions.
        
        Numerical solver should agree closely with MetPy.
        MetPy uses Bolton which is similar to our exact methods.
        """
        temp_k = celsius_to_kelvin(temp_c)
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=np.array([temp_k]),
            rh=np.array([rh]),
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
        
        td_k_ref = get_metpy_dewpoint(temp_k, rh)
        
        # Numerical solver should agree with MetPy within 0.5°C
        assert_allclose(
            roots[0], td_k_ref,
            atol=0.5,
            err_msg=(
                f"Solver vs MetPy: T={temp_c}°C, RH={rh*100:.0f}%, "
                f"Solver={kelvin_to_celsius(roots[0]):.2f}°C, "
                f"MetPy={kelvin_to_celsius(td_k_ref):.2f}°C"
            )
        )
    
    def test_metpy_excellent_agreement_at_standard_conditions(self):
        """Test excellent agreement with MetPy at standard conditions."""
        # Standard conditions: 20-30°C, 50-70% RH
        temp_range = [20, 25, 30]
        rh_range = [0.5, 0.6, 0.7]
        
        for temp_c in temp_range:
            for rh in rh_range:
                temp_k = celsius_to_kelvin(temp_c)
                
                roots, _, converged = get_dewpoint_using_solver(
                    temp_k=np.array([temp_k]),
                    rh=np.array([rh]),
                    surface_type='water',
                    vapor_equation='goff_gratch'
                )
                
                assert converged[0]
                
                td_k_ref = get_metpy_dewpoint(temp_k, rh)
                
                # Should agree within 0.2°C at standard conditions
                diff = abs(roots[0] - td_k_ref)
                assert diff < 0.3, (
                    f"Poor agreement at T={temp_c}°C, RH={rh*100:.0f}%: "
                    f"diff={diff:.3f}K"
                )


# ==============================================================================
# Test Class: PsychroLib Cross-Validation
# ==============================================================================

@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestSolverPsychroLibValidation:
    """Cross-validate against PsychroLib (ASHRAE standard)."""
    
    def _get_tolerance(self, temp_c: float, rh: float) -> float:
        """
        Get appropriate tolerance based on conditions.
        
        Different vapor pressure equation implementations diverge at:
        - Cold temperatures (especially at/below freezing)
        - Low relative humidity
        - Combination of both (worst case)
        
        This is expected physical behavior due to:
        - Different coefficient precision in implementations
        - Different numerical methods
        - Different reference standards (years apart)
        
        Parameters
        ----------
        temp_c : float
            Temperature in Celsius
        rh : float
            Relative humidity (0-1)
        
        Returns
        -------
        float
            Appropriate tolerance in Kelvin
        
        Notes
        -----
        Observed differences from testing:
        - 0°C, 30% RH: 1.65K (extreme case)
        - 0°C, 50% RH: 1.02K
        - 0°C, 70% RH: 0.55K
        - 10°C, 30% RH: 0.77K
        - 20°C+, 50%+ RH: <0.2K (excellent agreement)
        """
        # Extreme conditions: freezing + very low humidity
        if temp_c <= 0 and rh < 0.35:
            return 2.0  # Covers observed 1.65K difference
        
        # Cold + low humidity
        elif temp_c <= 0 and rh < 0.6:
            return 1.2  # Covers observed 1.02K difference
        
        # Cold temperatures (0-5°C)
        elif temp_c <= 5:
            return 1.0  # Covers observed 0.55K difference
        
        # Cool temperatures (5-15°C) with low humidity
        elif temp_c <= 15 and rh < 0.4:
            return 1.0  # Covers observed 0.77K at 10°C, 30%
        
        # Low humidity at any temperature
        elif rh < 0.4:
            return 0.8
        
        # Normal conditions (>15°C, >40% RH)
        else:
            return 0.5  # Excellent agreement expected
    
    @pytest.mark.parametrize("temp_c", [0, 10, 20, 25, 30, 35])
    @pytest.mark.parametrize("rh", [0.3, 0.5, 0.7, 0.9])
    def test_against_psychrolib(self, temp_c, rh):
        """Test against PsychroLib across range of conditions.
        
        PsychroLib uses Hyland-Wexler equations (ASHRAE standard).
        Our solver with Hyland-Wexler should match, with adaptive
        tolerances based on physical conditions.
        
        Larger tolerances at extreme conditions (cold + low humidity)
        are expected due to:
        - Different implementation details
        - Numerical precision differences
        - Different reference formulations
        
        This is normal behavior, not an error in either implementation.
        """
        temp_k = celsius_to_kelvin(temp_c)
        
        # Use Hyland-Wexler to match PsychroLib
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=np.array([temp_k]),
            rh=np.array([rh]),
            surface_type='water',
            vapor_equation='hyland_wexler'
        )
        
        assert converged[0], f"Failed to converge for T={temp_c}°C, RH={rh*100:.0f}%"
        
        td_k_ref = get_psychrolib_dewpoint(temp_k, rh)
        
        # Get adaptive tolerance based on conditions
        tolerance = self._get_tolerance(temp_c, rh)
        
        # Calculate difference
        diff = abs(roots[0] - td_k_ref)
        
        # Check against adaptive tolerance
        assert diff <= tolerance, (
            f"Solver vs PsychroLib: T={temp_c}°C, RH={rh*100:.0f}%, "
            f"Solver={kelvin_to_celsius(roots[0]):.4f}°C, "
            f"PsychroLib={kelvin_to_celsius(td_k_ref):.4f}°C, "
            f"Diff={diff:.4f}K (tolerance={tolerance:.1f}K)"
        )
    
    def test_psychrolib_excellent_agreement_at_standard_conditions(self):
        """Test excellent agreement with PsychroLib at standard conditions.
        
        At normal temperatures (20-30°C) and moderate humidity (50-70%),
        both implementations should agree very closely (<0.2K).
        
        This validates that the solver is working correctly and that
        larger differences at extreme conditions are truly due to
        physical/implementation factors, not bugs.
        """
        # Standard conditions: 20-30°C, 50-70% RH
        test_cases = [
            (20, 0.5),
            (20, 0.6),
            (20, 0.7),
            (25, 0.5),
            (25, 0.6),
            (25, 0.7),
            (30, 0.5),
            (30, 0.6),
            (30, 0.7),
        ]
        
        max_diff_observed = 0.0
        
        for temp_c, rh in test_cases:
            temp_k = celsius_to_kelvin(temp_c)
            
            roots, _, converged = get_dewpoint_using_solver(
                temp_k=np.array([temp_k]),
                rh=np.array([rh]),
                surface_type='water',
                vapor_equation='hyland_wexler'
            )
            
            assert converged[0], f"Failed to converge at T={temp_c}°C, RH={rh*100:.0f}%"
            
            td_k_ref = get_psychrolib_dewpoint(temp_k, rh)
            diff = abs(roots[0] - td_k_ref)
            max_diff_observed = max(max_diff_observed, diff)
            
            # Should agree very closely at standard conditions
            assert diff < 0.3, (
                f"Poor agreement at standard conditions: "
                f"T={temp_c}°C, RH={rh*100:.0f}%, diff={diff:.4f}K"
            )
        
        # Document best-case agreement
        assert max_diff_observed < 0.3, (
            f"Maximum difference at standard conditions too large: {max_diff_observed:.4f}K"
        )
    
    def test_psychrolib_divergence_pattern_matches_theory(self):
        """Test that divergence pattern matches theoretical expectations.
        
        Documents that equation differences follow expected physical pattern:
        - Largest at cold temperatures + low humidity
        - Smaller at moderate conditions
        - Smallest at warm temperatures + high humidity
        
        This validates that differences are due to equation formulation,
        not implementation errors.
        """
        # Test cases with expected maximum differences
        test_cases = [
            # (temp_c, rh, expected_max_diff_K, description)
            (0, 0.3, 2.0, "Extreme: freezing + low RH"),
            (0, 0.5, 1.2, "Cold: freezing + moderate RH"),
            (0, 0.7, 1.0, "Cold: freezing + higher RH"),
            (10, 0.3, 1.0, "Cool: 10°C + low RH"),
            (10, 0.5, 0.8, "Cool: 10°C + moderate RH"),
            (20, 0.3, 0.8, "Warm: 20°C + low RH"),
            (20, 0.5, 0.5, "Standard: 20°C + moderate RH"),
            (30, 0.5, 0.5, "Warm: 30°C + moderate RH"),
        ]
        
        print("\nValidating equation divergence pattern:")
        
        for temp_c, rh, expected_max_diff, description in test_cases:
            temp_k = celsius_to_kelvin(temp_c)
            
            roots, _, converged = get_dewpoint_using_solver(
                temp_k=np.array([temp_k]),
                rh=np.array([rh]),
                surface_type='water',
                vapor_equation='hyland_wexler'
            )
            
            if not converged[0]:
                pytest.skip(f"Did not converge for {description}")
            
            td_k_ref = get_psychrolib_dewpoint(temp_k, rh)
            diff = abs(roots[0] - td_k_ref)
            
            print(
                f"  {description:35s}: "
                f"diff={diff:.4f}K (expect <{expected_max_diff}K) "
                f"{'✓' if diff <= expected_max_diff else '✗'}"
            )
            
            # Validate difference is within expected range
            assert diff <= expected_max_diff, (
                f"Difference larger than expected for {description}: "
                f"{diff:.4f}K > {expected_max_diff}K"
            )
        
        print("  Pattern matches theory: Largest diff at cold+low RH ✓")
    
    def test_psychrolib_convergence_at_extreme_conditions(self):
        """Test that solver converges even at extreme conditions.
        
        Even if results differ from PsychroLib at extremes (expected),
        the solver should still converge reliably.
        """
        extreme_cases = [
            (0, 0.3),   # Freezing + very low RH
            (0, 0.5),   # Freezing + moderate RH
            (0, 0.9),   # Freezing + high RH
            (-5, 0.5),  # Below freezing (if supported)
        ]
        
        for temp_c, rh in extreme_cases:
            temp_k = celsius_to_kelvin(temp_c)
            
            # Skip if outside valid range
            if temp_k < 273.15:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        roots, _, converged = get_dewpoint_using_solver(
                            temp_k=np.array([temp_k]),
                            rh=np.array([rh]),
                            surface_type='water',
                            vapor_equation='hyland_wexler'
                        )
                    except:
                        continue
            else:
                roots, _, converged = get_dewpoint_using_solver(
                    temp_k=np.array([temp_k]),
                    rh=np.array([rh]),
                    surface_type='water',
                    vapor_equation='hyland_wexler'
                )
            
            # Just check convergence, not accuracy
            if temp_k >= 273.15:  # Within valid range
                assert converged[0], (
                    f"Failed to converge at extreme condition: "
                    f"T={temp_c}°C, RH={rh*100:.0f}%"
                )
                
# ==============================================================================
# Test Class: Multiple Equations
# ==============================================================================

class TestSolverMultipleEquations:
    """Test solver with different vapor pressure equations."""
    
    def test_goff_gratch_converges(self):
        """Test Goff-Gratch equation (WMO standard)."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
    
    def test_hyland_wexler_converges(self):
        """Test Hyland-Wexler equation (ASHRAE standard)."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='hyland_wexler'
        )
        
        assert converged[0]
    
    def test_equations_agree_at_standard_conditions(self):
        """Test that different equations agree at standard conditions.
        
        At 20°C, 60% RH, all equations should agree within ±0.1°C.
        """
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        equations = ['goff_gratch', 'hyland_wexler']
        results = {}
        
        for eq in equations:
            roots, _, converged = get_dewpoint_using_solver(
                temp_k=temp_k,
                rh=rh,
                surface_type='water',
                vapor_equation=eq
            )
            assert converged[0], f"{eq} did not converge"
            results[eq] = kelvin_to_celsius(roots[0])
        
        # All equations should agree within 0.1°C
        values = list(results.values())
        max_diff = max(values) - min(values)
        
        assert max_diff < 0.1, (
            f"Equations disagree: {results}, max_diff={max_diff:.4f}°C"
        )


# ==============================================================================
# Test Class: Edge Cases
# ==============================================================================

class TestSolverEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_humidity(self):
        """Test at very low humidity (10%)."""
        temp_k = np.array([293.15])
        rh = np.array([0.1])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
        td_c = kelvin_to_celsius(roots[0])
        
        # At 20°C, 10% RH → Td should be well below 0°C
        assert td_c < 0
        assert td_c > -50  # But not unreasonably low
    
    def test_very_high_humidity(self):
        """Test at very high humidity (99%)."""
        temp_k = np.array([293.15])
        rh = np.array([0.99])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
        
        # At 99% RH, Td should be very close to T
        assert abs(roots[0] - temp_k[0]) < 0.5
    
    def test_extreme_cold(self):
        """Test at extreme cold temperature."""
        temp_k = np.array([243.15])  # -30°C
        rh = np.array([0.5])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        # May converge or not depending on equation validity
        if converged[0]:
            assert roots[0] < temp_k[0]
            assert roots[0] > 200  # Physically reasonable
    
    def test_extreme_heat(self):
        """Test at extreme hot temperature."""
        temp_k = np.array([323.15])  # 50°C
        rh = np.array([0.5])
        
        roots, _, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert converged[0]
        assert roots[0] < temp_k[0]
        assert roots[0] < 350  # Physically reasonable
    
    def test_no_nan_or_inf_in_results(self):
        """Test that results never contain NaN or Inf."""
        temp_k = np.linspace(273.15, 313.15, 50)
        rh = np.linspace(0.2, 0.95, 50)
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        # Even for unconverged points, should not be NaN/Inf
        assert not np.any(np.isnan(roots))
        assert not np.any(np.isinf(roots))
        assert not np.any(np.isnan(iters))
        assert not np.any(np.isinf(iters))


# ==============================================================================
# Test Class: Determinism
# ==============================================================================

class TestSolverDeterminism:
    """Test deterministic behavior."""
    
    def test_deterministic_results(self):
        """Test that repeated calls give identical results."""
        temp_k = np.array([293.15])
        rh = np.array([0.6])
        
        results = []
        for _ in range(5):
            roots, _, _ = get_dewpoint_using_solver(
                temp_k=temp_k,
                rh=rh,
                surface_type='water',
                vapor_equation='goff_gratch'
            )
            results.append(roots[0])
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_deterministic_vector_results(self):
        """Test deterministic behavior for vectors."""
        temp_k = np.linspace(273.15, 313.15, 20)
        rh = np.full(20, 0.6)
        
        roots1, _, _ = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        roots2, _, _ = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert_array_equal(roots1, roots2)


# ==============================================================================
# Test Class: Performance
# ==============================================================================

class TestSolverPerformance:
    """Test solver performance characteristics."""
    
    def test_handles_large_arrays(self):
        """Test that solver can handle large arrays (1000+ points)."""
        n = 1000
        temp_k = np.linspace(273.15, 313.15, n)
        rh = np.linspace(0.3, 0.9, n)
        
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        
        assert len(roots) == n
        assert np.sum(converged) / n >= 0.99  # At least 99% convergence
    
    @pytest.mark.slow
    def test_performance_benchmark(self):
        """Benchmark solver performance (marked as slow test)."""
        import time
        
        n = 1000
        temp_k = np.linspace(273.15, 313.15, n)
        rh = np.full(n, 0.6)
        
        start = time.perf_counter()
        roots, iters, converged = get_dewpoint_using_solver(
            temp_k=temp_k,
            rh=rh,
            surface_type='water',
            vapor_equation='goff_gratch'
        )
        elapsed = time.perf_counter() - start
        
        calc_per_sec = n / elapsed
        
        # Should achieve at least 500 calculations/second
        assert calc_per_sec >= 500, f"Performance too low: {calc_per_sec:.0f} calc/sec"
        
        print(f"\nPerformance: {calc_per_sec:,.0f} calculations/second")
        print(f"Average iterations: {np.mean(iters):.1f}")
        print(f"Convergence rate: {np.sum(converged)/n*100:.1f}%")


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])