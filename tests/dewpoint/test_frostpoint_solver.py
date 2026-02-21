"""
Comprehensive test suite for Dewpoint.get_frostpoint_solver()

Tests exact frost point solver against reference implementations
from MetPy and PsychroLib to ensure accuracy and correctness.
"""

import warnings

import numpy as np
import pytest

# Import the function to test
from meteocalc.dewpoint.core import Dewpoint
from meteocalc.vapor._enums import VaporEquationName

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
        "temp_k": 263.15,  # -10°C
        "rh": 0.7,
        "expected_tf_c": -13.96,
    }


@pytest.fixture
def array_conditions():
    """Standard array test conditions."""
    return {
        "temps_k": np.array([253.15, 263.15, 268.15]),  # -20, -10, -5°C
        "rhs": np.array([0.5, 0.7, 0.9]),
        "expected_tfs_c": np.array([-28.35, -13.96, -6.05]),
    }


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestBasicFunctionality:
    """Test basic function behavior and return types."""

    def test_scalar_input_returns_scalar(self, scalar_conditions):
        """Scalar inputs should return scalar float."""
        tf = Dewpoint.get_frostpoint_solver(
            temp_k=scalar_conditions["temp_k"], rh=scalar_conditions["rh"]
        )

        assert isinstance(tf, (float, np.floating)), f"Expected float, got {type(tf)}"
        assert np.isfinite(tf), "Result should be finite"
        assert tf > 0, "Temperature in Kelvin should be positive"

    def test_array_input_returns_array(self, array_conditions):
        """Array inputs should return ndarray with correct shape."""
        tf = Dewpoint.get_frostpoint_solver(temp_k=array_conditions["temps_k"], rh=0.7)

        assert isinstance(tf, np.ndarray), f"Expected ndarray, got {type(tf)}"
        assert tf.shape == array_conditions["temps_k"].shape
        assert np.all(np.isfinite(tf))

    def test_list_input_works(self):
        """Lists should be converted and work correctly."""
        temps = [253.15, 263.15, 268.15]
        tf = Dewpoint.get_frostpoint_solver(temp_k=temps, rh=0.7)

        assert isinstance(tf, np.ndarray)
        assert len(tf) == len(temps)
        assert np.all(np.isfinite(tf))


# ============================================================================
# MetPy Validation Tests
# ============================================================================


@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestMetPyValidation:
    """
    Validate against MetPy implementation.

    Note: MetPy calculates dew point (water), not frost point (ice).
    Below 0°C, frost point should be higher than dew point.
    """

    def test_against_metpy_scalar(self, scalar_conditions):
        """Compare with MetPy for scalar inputs (frost > dew)."""
        temp_k = scalar_conditions["temp_k"]
        rh = scalar_conditions["rh"]

        # Our frost point solver
        tf_ours = Dewpoint.get_frostpoint_solver(temp_k, rh)

        # MetPy dew point (water surface)
        temp_metpy = (temp_k * units.K).to("degC")
        rh_metpy = rh * 100 * units.percent
        td_metpy = dewpoint_from_relative_humidity(temp_metpy, rh_metpy)
        td_metpy_k = td_metpy.to("K").magnitude

        # Frost point should be higher than dew point below 0°C
        assert tf_ours > td_metpy_k, (
            f"Frost point ({tf_ours}K) should be > dew point ({td_metpy_k}K)"
        )

        # Difference should be reasonable (0.1-5K typically)
        diff = tf_ours - td_metpy_k
        assert 0.1 < diff < 5.0, (
            f"Frost-dew difference {diff:.2f}K outside expected range"
        )

    def test_against_metpy_array(self, array_conditions):
        """Compare with MetPy for array inputs."""
        temps_k = array_conditions["temps_k"]
        rhs = array_conditions["rhs"]

        # Our frost point solver
        tf_ours = Dewpoint.get_frostpoint_solver(temps_k, rhs)

        # MetPy dew point element-wise
        for i, (temp_k, rh) in enumerate(zip(temps_k, rhs)):
            temp_metpy = (temp_k * units.K).to("degC")
            rh_metpy = rh * 100 * units.percent
            td_metpy = dewpoint_from_relative_humidity(temp_metpy, rh_metpy)
            td_metpy_k = td_metpy.to("K").magnitude

            # Frost point should be higher
            assert tf_ours[i] > td_metpy_k, (
                f"Index {i}: Frost point should be > dew point"
            )

            # Reasonable difference
            diff = tf_ours[i] - td_metpy_k
            assert 0.1 < diff < 5.0, (
                f"Index {i}: difference {diff:.2f}K outside expected range"
            )

    @pytest.mark.parametrize(
        "temp_c,rh",
        [
            (-10.0, 0.7),
            (-20.0, 0.5),
            (-5.0, 0.9),
            (-15.0, 0.6),
        ],
    )
    def test_against_metpy_parametrized(self, temp_c, rh):
        """Parametrized comparison with MetPy (frost > dew)."""
        temp_k = temp_c + 273.15

        # Our frost point
        tf_ours = Dewpoint.get_frostpoint_solver(temp_k, rh)

        # MetPy dew point
        td_metpy = (
            dewpoint_from_relative_humidity(
                (temp_k * units.K).to("degC"), rh * 100 * units.percent
            )
            .to("K")
            .magnitude
        )

        # Frost point should be higher
        assert tf_ours > td_metpy, (
            f"At {temp_c}°C, {rh * 100:.0f}% RH:\n"
            f"  Frost point: {tf_ours:.4f}K\n"
            f"  Dew point:   {td_metpy:.4f}K"
        )


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
        temp_k = scalar_conditions["temp_k"]
        rh = scalar_conditions["rh"]
        temp_c = temp_k - 273.15

        # Our implementation
        tf_ours_c = Dewpoint.get_frostpoint_solver(temp_k, rh) - 273.15

        # PsychroLib implementation (dew point, not frost point)
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        # Note: PsychroLib gives dew point, we calculate frost point
        # Frost point should be higher than dew point
        assert tf_ours_c > td_psychro_c, (
            f"Frost point ({tf_ours_c:.4f}°C) should be > dew point ({td_psychro_c:.4f}°C)"
        )

        # Difference should be reasonable
        diff = abs(tf_ours_c - td_psychro_c)
        assert diff < 5.0, (
            f"PsychroLib validation failed:\n"
            f"  Ours (frost): {tf_ours_c:.4f}°C\n"
            f"  PsychroLib (dew): {td_psychro_c:.4f}°C\n"
            f"  Difference: {diff:.4f}°C"
        )

    def test_against_psychrolib_array(self, array_conditions):
        """Compare with PsychroLib for array inputs."""
        temps_k = array_conditions["temps_k"]
        rhs = array_conditions["rhs"]

        # Our implementation
        tf_ours_c = Dewpoint.get_frostpoint_solver(temps_k, rhs) - 273.15

        # PsychroLib element-wise
        td_psychro_c = np.array(
            [psy.GetTDewPointFromRelHum(t - 273.15, r) for t, r in zip(temps_k, rhs)]
        )

        # Frost point should be higher than dew point
        assert np.all(tf_ours_c > td_psychro_c), (
            "All frost points should be > dew points"
        )

        # Differences should be reasonable
        diffs = tf_ours_c - td_psychro_c
        assert np.all(diffs < 5.0), (
            f"PsychroLib array validation:\n"
            f"  Ours (frost):      {np.round(tf_ours_c, 4)}\n"
            f"  PsychroLib (dew):  {np.round(td_psychro_c, 4)}\n"
            f"  Differences:       {np.round(diffs, 4)}"
        )

    @pytest.mark.parametrize(
        "temp_c,rh",
        [
            (-10.0, 0.7),
            (-20.0, 0.5),
            (-5.0, 0.9),
            (-15.0, 0.6),
        ],
    )
    def test_against_psychrolib_parametrized(self, temp_c, rh):
        """Parametrized comparison with PsychroLib."""
        temp_k = temp_c + 273.15

        tf_ours_c = Dewpoint.get_frostpoint_solver(temp_k, rh) - 273.15
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        # Frost point should be higher
        assert tf_ours_c > td_psychro_c, (
            f"At {temp_c}°C, {rh * 100:.0f}% RH:\n"
            f"  Frost point: {tf_ours_c:.4f}°C\n"
            f"  Dew point:   {td_psychro_c:.4f}°C"
        )


# ============================================================================
# Cross-Reference Validation
# ============================================================================


@pytest.mark.skipif(
    not (HAS_METPY and HAS_PSYCHROLIB), reason="Both MetPy and PsychroLib required"
)
class TestCrossReference:
    """
    Cross-validate between all three implementations.

    Note: MetPy and PsychroLib calculate dew point (water), not frost point.
    Below 0°C, frost point should be higher than dew point.
    """

    @pytest.mark.parametrize(
        "temp_c,rh",
        [
            (-10.0, 0.7),
            (-20.0, 0.5),
            (-5.0, 0.9),
            (-15.0, 0.6),
        ],
    )
    def test_all_three_physical_relationship(self, temp_c, rh):
        """
        Verify physical relationship: frost point > dew point below 0°C.
        """
        temp_k = temp_c + 273.15

        # Our frost point solver
        tf_ours = Dewpoint.get_frostpoint_solver(temp_k, rh) - 273.15

        # MetPy dew point
        td_metpy = dewpoint_from_relative_humidity(
            (temp_k * units.K).to("degC"), rh * 100 * units.percent
        ).magnitude

        # PsychroLib dew point
        td_psychro = psy.GetTDewPointFromRelHum(temp_c, rh)

        # Our frost point should be higher than both dew points
        assert tf_ours > td_metpy, (
            f"Frost point ({tf_ours:.2f}°C) should be > MetPy dew point ({td_metpy:.2f}°C)"
        )

        assert tf_ours > td_psychro, (
            f"Frost point ({tf_ours:.2f}°C) should be > PsychroLib dew point ({td_psychro:.2f}°C)"
        )

        # MetPy and PsychroLib dew points should be similar
        dew_diff = abs(td_metpy - td_psychro)
        assert dew_diff < 1.5, (
            f"MetPy and PsychroLib dew points differ by {dew_diff:.2f}°C"
        )

    def test_all_three_agree_on_trend(self):
        """All three should show same trend with increasing RH."""
        temp_k = 263.15
        rhs = np.array([0.3, 0.5, 0.7, 0.9])

        # Our frost point
        tf_ours = Dewpoint.get_frostpoint_solver(temp_k, rhs) - 273.15

        # MetPy dew point
        td_metpy = np.array(
            [
                dewpoint_from_relative_humidity(
                    (temp_k * units.K).to("degC"), rh * 100 * units.percent
                ).magnitude
                for rh in rhs
            ]
        )

        # PsychroLib dew point
        td_psychro = np.array(
            [psy.GetTDewPointFromRelHum(temp_k - 273.15, rh) for rh in rhs]
        )

        # All should increase monotonically with RH
        assert np.all(np.diff(tf_ours) > 0), "Ours: frost point should increase with RH"
        assert np.all(np.diff(td_metpy) > 0), "MetPy: dew point should increase with RH"
        assert np.all(np.diff(td_psychro) > 0), (
            "PsychroLib: dew point should increase with RH"
        )


# ============================================================================
# Accuracy Tests (Solver vs Approximation)
# ============================================================================


class TestAccuracy:
    """Test that exact solver is more accurate than approximations."""

    def test_solver_more_accurate_than_approximation(self):
        """Exact solver should be much more accurate than approximation."""
        temp_k = 263.15
        rh = 0.7
        noaa_expected = -13.96  # °C

        # Exact solver
        tf_solver = Dewpoint.get_frostpoint_solver(temp_k, rh) - 273.15

        # Approximation
        tf_approx = Dewpoint.get_frostpoint_approximation(temp_k, rh) - 273.15

        # Solver should be much closer to NOAA than approximation
        solver_error = abs(tf_solver - noaa_expected)
        approx_error = abs(tf_approx - noaa_expected)

        assert solver_error < approx_error, (
            f"Solver should be more accurate:\n"
            f"  Solver error: {solver_error:.4f}°C\n"
            f"  Approx error: {approx_error:.4f}°C"
        )

        # Solver should be within ±0.01°C
        assert solver_error < 0.01, f"Solver error {solver_error:.4f}°C exceeds ±0.01°C"


# ============================================================================
# Equation Selection Tests
# ============================================================================


class TestEquationSelection:
    """Test different vapor pressure equations."""

    def test_goff_gratch_default(self):
        """Goff-Gratch should be the default equation."""
        tf_default = Dewpoint.get_frostpoint_solver(263.15, 0.7)
        tf_goff = Dewpoint.get_frostpoint_solver(
            263.15, 0.7, vapor_equation_name="goff_gratch"
        )

        assert tf_default == tf_goff

    def test_enum_input_works(self):
        """Enum input should work same as string."""
        tf_string = Dewpoint.get_frostpoint_solver(263.15, 0.7, "goff_gratch")
        tf_enum = Dewpoint.get_frostpoint_solver(
            263.15, 0.7, VaporEquationName.GOFF_GRATCH
        )

        assert tf_string == tf_enum


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_equation_raises(self):
        """Invalid equation name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid enum"):
            Dewpoint.get_frostpoint_solver(263.15, 0.7, vapor_equation_name="invalid")

    def test_rh_below_zero_raises(self):
        """Negative RH should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_frostpoint_solver(263.15, -0.1)

    def test_rh_above_one_raises(self):
        """RH > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_frostpoint_solver(263.15, 1.5)


# ============================================================================
# Broadcasting Tests
# ============================================================================


class TestBroadcasting:
    """Test broadcasting between scalar and array inputs."""

    def test_scalar_temp_array_rh(self):
        """Scalar temp with array RH should broadcast."""
        rhs = np.array([0.5, 0.6, 0.7, 0.8])
        tf = Dewpoint.get_frostpoint_solver(temp_k=263.15, rh=rhs)

        assert tf.shape == rhs.shape
        assert np.all(np.isfinite(tf))

    def test_array_temp_scalar_rh(self):
        """Array temp with scalar RH should broadcast."""
        temps = np.array([253.15, 263.15, 268.15])
        tf = Dewpoint.get_frostpoint_solver(temp_k=temps, rh=0.7)

        assert tf.shape == temps.shape
        assert np.all(np.isfinite(tf))


# ============================================================================
# Physical Correctness Tests
# ============================================================================


class TestPhysicalCorrectness:
    """Test physical relationships and constraints."""

    def test_frost_point_below_air_temp(self):
        """Frost point must be at or below air temperature."""
        temp_k = 263.15
        rh = 0.7
        tf = Dewpoint.get_frostpoint_solver(temp_k, rh)

        assert tf <= temp_k, f"Frost point ({tf}K) must be <= air temp ({temp_k}K)"

    def test_saturation_equals_temp(self):
        """At 100% RH, frost point should equal air temperature."""
        temp_k = 263.15
        tf = Dewpoint.get_frostpoint_solver(temp_k, 1.0)

        # Exact solver should be very close (within 0.001K)
        assert abs(tf - temp_k) < 0.001, (
            f"At saturation, frost point ({tf}K) should equal temp ({temp_k}K)"
        )

    def test_frost_above_dew_below_freezing(self):
        """Below 0°C, frost point should be higher than dew point."""
        temp_k = 263.15
        rh = 0.7

        tf = Dewpoint.get_frostpoint_solver(temp_k, rh)
        td = Dewpoint.get_dewpoint_solver(temp_k, rh)

        assert tf > td, f"Frost point ({tf}K) should be > dew point ({td}K)"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with other meteocalc components."""

    def test_solver_vs_approximation_consistency(self):
        """Solver and approximation should be close but solver more accurate."""
        temp_k = 263.15
        rh = 0.7

        tf_solver = Dewpoint.get_frostpoint_solver(temp_k, rh)
        tf_approx = Dewpoint.get_frostpoint_approximation(temp_k, rh)

        # Should be close (within ~0.5K)
        diff = abs(tf_solver - tf_approx)
        assert diff < 0.5, f"Solver and approximation differ by {diff:.3f}K"

    def test_convenience_wrapper_consistent(self):
        """Convenience wrapper should give same result as explicit call."""
        temp_k = 263.15
        rh = 0.7

        tf_explicit = Dewpoint.get_frostpoint_solver(temp_k, rh)
        tf_wrapper = Dewpoint.get_frostpoint(temp_k, rh, calculation_method="solver")

        assert tf_explicit == tf_wrapper


# ============================================================================
# Summary Report
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
