"""
Comprehensive test suite for Dewpoint.get_dewpoint_solver()

Tests exact dew point solver against reference implementations
from MetPy and PsychroLib to ensure accuracy and correctness.
"""

import pytest
import numpy as np
import warnings

# Import the function to test
from meteocalc.dewpoint.core import Dewpoint
from meteocalc.vapor._enums import EquationName

# Import reference implementations
try:
    from metpy.calc import dewpoint_from_relative_humidity
    from metpy.units import units
    HAS_METPY = True
except ImportError:
    HAS_METPY = False
    warnings.warn("MetPy not installed - skipping MetPy validation tests")

try:
    import psychrolib as psy
    psy.SetUnitSystem(psy.SI)
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False
    warnings.warn("PsychroLib not installed - skipping PsychroLib validation tests")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def scalar_conditions():
    """Standard scalar test conditions."""
    return {
        'temp_k': 293.15,  # 20°C
        'rh': 0.6,
        'expected_td_c': 12.0,
    }


@pytest.fixture
def array_conditions():
    """Standard array test conditions."""
    return {
        'temps_k': np.array([273.15, 283.15, 293.15, 303.15]),  # 0, 10, 20, 30°C
        'rhs': np.array([0.5, 0.6, 0.7, 0.8]),
        'expected_tds_c': np.array([-9.18, 2.60, 14.37, 26.17]),
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestBasicFunctionality:
    """Test basic function behavior and return types."""

    def test_scalar_input_returns_scalar(self, scalar_conditions):
        """Scalar inputs should return scalar float."""
        td = Dewpoint.get_dewpoint_solver(
            temp_k=scalar_conditions['temp_k'],
            rh=scalar_conditions['rh']
        )

        assert isinstance(td, (float, np.floating)), \
            f"Expected float, got {type(td)}"
        assert np.isfinite(td), "Result should be finite"
        assert td > 0, "Temperature in Kelvin should be positive"

    def test_array_input_returns_array(self, array_conditions):
        """Array inputs should return ndarray with correct shape."""
        td = Dewpoint.get_dewpoint_solver(
            temp_k=array_conditions['temps_k'],
            rh=0.6
        )

        assert isinstance(td, np.ndarray), \
            f"Expected ndarray, got {type(td)}"
        assert td.shape == array_conditions['temps_k'].shape
        assert np.all(np.isfinite(td))

    def test_list_input_works(self):
        """Lists should be converted and work correctly."""
        temps = [273.15, 283.15, 293.15]
        td = Dewpoint.get_dewpoint_solver(temp_k=temps, rh=0.6)

        assert isinstance(td, np.ndarray)
        assert len(td) == len(temps)
        assert np.all(np.isfinite(td))

# ============================================================================
# MetPy Validation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestMetPyValidation:
    """Validate against MetPy implementation."""

    def test_against_metpy_scalar(self, scalar_conditions):
        """Compare with MetPy for scalar input."""
        temp_k = scalar_conditions['temp_k']
        rh = scalar_conditions['rh']

        # Our implementation
        td_ours_k = Dewpoint.get_dewpoint_solver(temp_k, rh)
        td_ours_c = td_ours_k - 273.15

        # MetPy implementation
        temp_metpy = (temp_k * units.K).to('degC')
        rh_metpy = rh * 100 * units.percent
        td_metpy_c = dewpoint_from_relative_humidity(
            temp_metpy, rh_metpy
        ).magnitude

        error = abs(td_ours_c - td_metpy_c)
        assert error < 0.5, \
            f"MetPy validation failed:\n" \
            f"  Ours:  {td_ours_c:.4f}°C\n" \
            f"  MetPy: {td_metpy_c:.4f}°C\n" \
            f"  Error: {error:.4f}°C"

    def test_against_metpy_array(self, array_conditions):
        """Compare with MetPy for array inputs."""
        temps_k = array_conditions['temps_k']
        rhs = array_conditions['rhs']

        # Our implementation
        td_ours_k = Dewpoint.get_dewpoint_solver(temps_k, rhs)
        td_ours_c = td_ours_k - 273.15

        # MetPy implementation element-wise
        td_metpy_c = np.array([
            dewpoint_from_relative_humidity(
                (t * units.K).to('degC'),
                r * 100 * units.percent
            ).magnitude
            for t, r in zip(temps_k, rhs)
        ])

        errors = np.abs(td_ours_c - td_metpy_c)
        max_error = np.max(errors)

        assert max_error < 0.5, \
            f"MetPy array validation failed:\n" \
            f"  Ours:      {np.round(td_ours_c, 4)}\n" \
            f"  MetPy:     {np.round(td_metpy_c, 4)}\n" \
            f"  Max error: {max_error:.4f}°C"

    @pytest.mark.parametrize("temp_c,rh", [
        (20.0, 0.6),
        (30.0, 0.8),
        (10.0, 0.5),
        (0.0, 0.7),
        (25.0, 0.4),
    ])
    def test_against_metpy_parametrized(self, temp_c, rh):
        """Parametrized comparison with MetPy."""
        temp_k = temp_c + 273.15

        td_ours_c = Dewpoint.get_dewpoint_solver(temp_k, rh) - 273.15

        td_metpy_c = dewpoint_from_relative_humidity(
            (temp_k * units.K).to('degC'),
            rh * 100 * units.percent
        ).magnitude

        error = abs(td_ours_c - td_metpy_c)
        assert error < 0.5, \
            f"At {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  Ours:  {td_ours_c:.4f}°C\n" \
            f"  MetPy: {td_metpy_c:.4f}°C\n" \
            f"  Error: {error:.4f}°C"


# ============================================================================
# PsychroLib Validation Tests
# ============================================================================

@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestPsychroLibValidation:
    """
    Validate against PsychroLib implementation.
    
    Note: PsychroLib uses ASHRAE formulas which may differ from our
    implementation by up to 1.5°C at boundary conditions.
    """

    PSYCHROLIB_TOLERANCE = 1.5

    def test_against_psychrolib_scalar(self, scalar_conditions):
        """Compare with PsychroLib for scalar input."""
        temp_k = scalar_conditions['temp_k']
        rh = scalar_conditions['rh']
        temp_c = temp_k - 273.15

        # Our implementation
        td_ours_c = Dewpoint.get_dewpoint_solver(temp_k, rh) - 273.15

        # PsychroLib implementation
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        error = abs(td_ours_c - td_psychro_c)
        assert error < self.PSYCHROLIB_TOLERANCE, \
            f"PsychroLib validation failed:\n" \
            f"  Ours:       {td_ours_c:.4f}°C\n" \
            f"  PsychroLib: {td_psychro_c:.4f}°C\n" \
            f"  Error:      {error:.4f}°C\n" \
            f"  Tolerance:  {self.PSYCHROLIB_TOLERANCE}°C\n" \
            f"  Note: Different vapor pressure formulas can differ at boundary conditions"

    def test_against_psychrolib_array(self, array_conditions):
        """Compare with PsychroLib for array inputs."""
        temps_k = array_conditions['temps_k']
        rhs = array_conditions['rhs']

        # Our implementation
        td_ours_c = Dewpoint.get_dewpoint_solver(temps_k, rhs) - 273.15

        # PsychroLib element-wise
        td_psychro_c = np.array([
            psy.GetTDewPointFromRelHum(t - 273.15, r)
            for t, r in zip(temps_k, rhs)
        ])

        errors = np.abs(td_ours_c - td_psychro_c)
        max_error = np.max(errors)

        assert max_error < self.PSYCHROLIB_TOLERANCE, \
            f"PsychroLib array validation failed:\n" \
            f"  Ours:       {np.round(td_ours_c, 4)}\n" \
            f"  PsychroLib: {np.round(td_psychro_c, 4)}\n" \
            f"  Errors:     {np.round(errors, 4)}\n" \
            f"  Max error:  {max_error:.4f}°C\n" \
            f"  Tolerance:  {self.PSYCHROLIB_TOLERANCE}°C"

    @pytest.mark.parametrize("temp_c,rh,tolerance", [
        (20.0, 0.6, 0.5),   # Standard: formulas agree well
        (30.0, 0.8, 0.5),   # Hot and humid: formulas agree well
        (10.0, 0.5, 0.5),   # Cool: formulas agree well
        (0.0, 0.7, 1.5),    # Phase boundary: larger tolerance needed
        (25.0, 0.4, 0.5),   # Warm dry: formulas agree well
    ])
    def test_against_psychrolib_parametrized(self, temp_c, rh, tolerance):
        """
        Parametrized comparison with PsychroLib.
        
        Note: 0°C uses larger tolerance (1.5°C) because it is a phase
        boundary where different formulas may diverge more.
        """
        temp_k = temp_c + 273.15

        td_ours_c = Dewpoint.get_dewpoint_solver(temp_k, rh) - 273.15
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        error = abs(td_ours_c - td_psychro_c)
        assert error < tolerance, \
            f"At {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  Ours:       {td_ours_c:.4f}°C\n" \
            f"  PsychroLib: {td_psychro_c:.4f}°C\n" \
            f"  Error:      {error:.4f}°C\n" \
            f"  Tolerance:  {tolerance}°C"


# ============================================================================
# Cross-Reference Validation
# ============================================================================

@pytest.mark.skipif(
    not (HAS_METPY and HAS_PSYCHROLIB),
    reason="Both MetPy and PsychroLib required"
)
class TestCrossReference:
    """Cross-validate between all three implementations."""

    METPY_TOLERANCE = 0.5       # Our solver vs MetPy (similar methods)
    PSYCHROLIB_TOLERANCE = 1.5  # Different formulas can differ more

    @pytest.mark.parametrize("temp_c,rh", [
        (20.0, 0.6),
        (30.0, 0.8),
        (10.0, 0.5),
        (0.0, 0.7),
        (25.0, 0.4),
    ])
    def test_all_three_agree(self, temp_c, rh):
        """All three implementations should agree within formula tolerances."""
        temp_k = temp_c + 273.15

        # Our implementation
        td_ours_c = Dewpoint.get_dewpoint_solver(temp_k, rh) - 273.15

        # MetPy
        td_metpy_c = dewpoint_from_relative_humidity(
            (temp_k * units.K).to('degC'),
            rh * 100 * units.percent
        ).magnitude

        # PsychroLib
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        # Ours vs MetPy (should be close)
        ours_metpy_diff = abs(td_ours_c - td_metpy_c)
        assert ours_metpy_diff < self.METPY_TOLERANCE, \
            f"Ours vs MetPy at {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  Ours:  {td_ours_c:.4f}°C\n" \
            f"  MetPy: {td_metpy_c:.4f}°C\n" \
            f"  Diff:  {ours_metpy_diff:.4f}°C"

        # Ours vs PsychroLib (different formulas, can differ more)
        ours_psychro_diff = abs(td_ours_c - td_psychro_c)
        assert ours_psychro_diff < self.PSYCHROLIB_TOLERANCE, \
            f"Ours vs PsychroLib at {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  Ours:       {td_ours_c:.4f}°C\n" \
            f"  PsychroLib: {td_psychro_c:.4f}°C\n" \
            f"  Diff:       {ours_psychro_diff:.4f}°C"

        # MetPy vs PsychroLib (same formula difference)
        metpy_psychro_diff = abs(td_metpy_c - td_psychro_c)
        assert metpy_psychro_diff < self.PSYCHROLIB_TOLERANCE, \
            f"MetPy vs PsychroLib at {temp_c}°C, {rh*100:.0f}% RH:\n" \
            f"  MetPy:      {td_metpy_c:.4f}°C\n" \
            f"  PsychroLib: {td_psychro_c:.4f}°C\n" \
            f"  Diff:       {metpy_psychro_diff:.4f}°C"

    def test_all_three_agree_on_trend(self):
        """All three should show same trend with increasing RH."""
        temp_k = 293.15
        rhs = np.array([0.3, 0.5, 0.7, 0.9])

        # Our implementation
        td_ours = Dewpoint.get_dewpoint_solver(temp_k, rhs) - 273.15

        # MetPy
        td_metpy = np.array([
            dewpoint_from_relative_humidity(
                (temp_k * units.K).to('degC'),
                rh * 100 * units.percent
            ).magnitude
            for rh in rhs
        ])

        # PsychroLib
        td_psychro = np.array([
            psy.GetTDewPointFromRelHum(temp_k - 273.15, rh)
            for rh in rhs
        ])

        # All should increase monotonically with RH
        assert np.all(np.diff(td_ours) > 0), \
            "Ours: dew point should increase with RH"
        assert np.all(np.diff(td_metpy) > 0), \
            "MetPy: dew point should increase with RH"
        assert np.all(np.diff(td_psychro) > 0), \
            "PsychroLib: dew point should increase with RH"

# ============================================================================
# Equation Selection Tests
# ============================================================================

class TestEquationSelection:
    """Test different vapor pressure equations."""

    def test_goff_gratch_default(self):
        """Goff-Gratch should be the default equation."""
        td_default = Dewpoint.get_dewpoint_solver(293.15, 0.6)
        td_goff = Dewpoint.get_dewpoint_solver(
            293.15, 0.6, vapor_equation_name='goff_gratch'
        )

        assert td_default == td_goff

    def test_enum_input_works(self):
        """Enum input should work same as string."""
        td_string = Dewpoint.get_dewpoint_solver(293.15, 0.6, 'goff_gratch')
        td_enum = Dewpoint.get_dewpoint_solver(
            293.15, 0.6, EquationName.GOFF_GRATCH
        )

        assert td_string == td_enum


# ============================================================================
# Input Validation Tests
# ============================================================================

class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_equation_raises(self):
        """Invalid equation name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid enum"):
            Dewpoint.get_dewpoint_solver(
                293.15, 0.6, vapor_equation_name='invalid'
            )

    def test_rh_below_zero_raises(self):
        """Negative RH should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_dewpoint_solver(293.15, -0.1)

    def test_rh_above_one_raises(self):
        """RH > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_dewpoint_solver(293.15, 1.5)


# ============================================================================
# Broadcasting Tests
# ============================================================================

class TestBroadcasting:
    """Test broadcasting between scalar and array inputs."""

    def test_scalar_temp_array_rh(self):
        """Scalar temp with array RH should broadcast."""
        rhs = np.array([0.5, 0.6, 0.7, 0.8])
        td = Dewpoint.get_dewpoint_solver(temp_k=293.15, rh=rhs)

        assert td.shape == rhs.shape
        assert np.all(np.isfinite(td))

    def test_array_temp_scalar_rh(self):
        """Array temp with scalar RH should broadcast."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        td = Dewpoint.get_dewpoint_solver(temp_k=temps, rh=0.6)

        assert td.shape == temps.shape
        assert np.all(np.isfinite(td))


# ============================================================================
# Physical Correctness Tests
# ============================================================================

class TestPhysicalCorrectness:
    """Test physical relationships and constraints."""

    def test_dewpoint_below_air_temp(self):
        """Dew point must be at or below air temperature."""
        temp_k = 293.15
        td = Dewpoint.get_dewpoint_solver(temp_k, 0.6)

        assert td <= temp_k, \
            f"Dew point ({td}K) must be <= air temp ({temp_k}K)"

    def test_saturation_equals_temp(self):
        """At 100% RH, dew point should equal air temperature."""
        temp_k = 293.15
        td = Dewpoint.get_dewpoint_solver(temp_k, 1.0)

        # Exact solver should be very close (within 0.001K)
        assert abs(td - temp_k) < 0.001, \
            f"At saturation, dew point ({td}K) should equal temp ({temp_k}K)"

    def test_dew_below_frost_below_freezing(self):
        """Below 0°C, dew point should be lower than frost point."""
        temp_k = 263.15  # -10°C
        rh = 0.7

        td = Dewpoint.get_dewpoint_solver(temp_k, rh)
        tf = Dewpoint.get_frostpoint_solver(temp_k, rh)

        assert td < tf, \
            f"Dew point ({td}K) should be < frost point ({tf}K)"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with other meteocalc components."""

    def test_solver_vs_approximation_consistency(self):
        """Solver and approximation should be close but solver more accurate."""
        temp_k = 293.15
        rh = 0.6

        td_solver = Dewpoint.get_dewpoint_solver(temp_k, rh)
        td_approx = Dewpoint.get_dewpoint_approximation(temp_k, rh)

        # Should be close (within ~0.5K)
        diff = abs(td_solver - td_approx)
        assert diff < 0.5, \
            f"Solver and approximation differ by {diff:.3f}K"

    def test_convenience_wrapper_consistent(self):
        """Convenience wrapper should give same result as explicit call."""
        temp_k = 293.15
        rh = 0.6

        td_explicit = Dewpoint.get_dewpoint_solver(temp_k, rh)
        td_wrapper = Dewpoint.get_dewpoint(
            temp_k, rh, calculation_method='solver'
        )

        assert td_explicit == td_wrapper


# ============================================================================
# Summary Report
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])