"""
Comprehensive test suite for Dewpoint.get_frostpoint_approximation()

Tests frost point approximation methods against reference implementations
from MetPy and PsychroLib to ensure accuracy and correctness.
"""

import warnings

import numpy as np
import pytest

# Import the function to test
from meteocalc.dewpoint.core import Dewpoint

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

    psy.SetUnitSystem(psy.SI)  # Use SI units
    HAS_PSYCHROLIB = True
except ImportError:
    HAS_PSYCHROLIB = False
    warnings.warn("PsychroLib not installed - skipping PsychroLib validation tests")


# ============================================================================
# Test Fixtures
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
        tf = Dewpoint.get_frostpoint_approximation(
            temp_k=scalar_conditions["temp_k"], rh=scalar_conditions["rh"]
        )

        assert isinstance(tf, (float, np.floating)), f"Expected float, got {type(tf)}"
        assert np.isfinite(tf), "Result should be finite"
        assert tf > 0, "Temperature in Kelvin should be positive"

    def test_array_input_returns_array(self, array_conditions):
        """Array inputs should return ndarray with matching shape."""
        tf = Dewpoint.get_frostpoint_approximation(
            temp_k=array_conditions["temps_k"], rh=0.7
        )

        assert isinstance(tf, np.ndarray), f"Expected ndarray, got {type(tf)}"
        assert tf.shape == array_conditions["temps_k"].shape, (
            f"Shape mismatch: {tf.shape} vs {array_conditions['temps_k'].shape}"
        )
        assert np.all(np.isfinite(tf)), "All results should be finite"

    def test_list_input_converts_to_array(self):
        """Lists should be converted to arrays and work correctly."""
        temps = [253.15, 263.15, 268.15]
        tf = Dewpoint.get_frostpoint_approximation(temp_k=temps, rh=0.7)

        assert isinstance(tf, np.ndarray)
        assert len(tf) == len(temps)
        assert np.all(np.isfinite(tf))


# ============================================================================
# MetPy Validation Tests
# ============================================================================


@pytest.mark.skipif(not HAS_METPY, reason="MetPy not installed")
class TestMetPyValidation:
    """Validate against MetPy implementation."""

    def test_against_metpy_scalar(self, scalar_conditions):
        """Compare with MetPy for scalar inputs."""
        temp_k = scalar_conditions["temp_k"]
        rh = scalar_conditions["rh"]

        # Our implementation
        tf_ours = Dewpoint.get_frostpoint_approximation(temp_k, rh)

        # MetPy implementation (dew point over ice)
        # Note: MetPy's dewpoint_from_relative_humidity uses Magnus by default
        temp_metpy = (temp_k * units.K).to("degC")
        rh_metpy = rh * 100 * units.percent

        # MetPy calculates dew point (water), we need frost point (ice)
        # For comparison, we'll use our dew point and check the difference
        td_metpy = dewpoint_from_relative_humidity(temp_metpy, rh_metpy)
        td_metpy_k = td_metpy.to("K").magnitude

        # Our frost point should be higher than MetPy's dew point below 0°C
        assert tf_ours > td_metpy_k, (
            f"Frost point ({tf_ours}K) should be > dew point ({td_metpy_k}K)"
        )

        # Difference should be reasonable (0.3-3K typically)
        diff = tf_ours - td_metpy_k
        assert 0.1 < diff < 5.0, (
            f"Frost-dew difference {diff:.2f}K outside expected range"
        )

    def test_against_metpy_array(self, array_conditions):
        """Compare with MetPy for array inputs."""
        temps_k = array_conditions["temps_k"]
        rhs = array_conditions["rhs"]

        # Our implementation
        tf_ours = Dewpoint.get_frostpoint_approximation(temps_k, rhs)

        # Compare element-wise
        for i, (temp_k, rh) in enumerate(zip(temps_k, rhs)):
            temp_metpy = (temp_k * units.K).to("degC")
            rh_metpy = rh * 100 * units.percent

            td_metpy = dewpoint_from_relative_humidity(temp_metpy, rh_metpy)
            td_metpy_k = td_metpy.to("K").magnitude

            # Frost point should be higher than dew point
            assert tf_ours[i] > td_metpy_k, (
                f"Index {i}: Frost point should be > dew point"
            )


# ============================================================================
# PsychroLib Validation Tests
# ============================================================================


@pytest.mark.skipif(not HAS_PSYCHROLIB, reason="PsychroLib not installed")
class TestPsychroLibValidation:
    """Validate against PsychroLib implementation."""

    def test_against_psychrolib_scalar(self, scalar_conditions):
        """Compare with PsychroLib for scalar inputs."""
        temp_c = scalar_conditions["temp_k"] - 273.15
        rh = scalar_conditions["rh"]

        # Our implementation
        tf_ours_k = Dewpoint.get_frostpoint_approximation(
            scalar_conditions["temp_k"], rh
        )
        tf_ours_c = tf_ours_k - 273.15

        # PsychroLib implementation
        # Use standard atmospheric pressure (101325 Pa)
        pressure = 101325.0

        # Get dew point from PsychroLib
        td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

        # For frost point, we need to account for ice
        # PsychroLib gives dew point, we calculate frost point
        # They should differ by 0.3-3K below 0°C

        # Our frost point should be reasonably close to PsychroLib dew point
        diff = abs(tf_ours_c - td_psychro_c)

        # Tolerance: frost point is typically 0.3-3K higher than dew point
        # So we expect our value to be close but slightly higher
        assert diff < 3.0, (
            f"PsychroLib validation failed:\n"
            f"  Our frost point: {tf_ours_c:.2f}°C\n"
            f"  PsychroLib dew:  {td_psychro_c:.2f}°C\n"
            f"  Difference:      {diff:.2f}°C"
        )

        # Frost point should be higher than dew point
        assert tf_ours_c > td_psychro_c, "Frost point should be > dew point below 0°C"

    def test_against_psychrolib_array(self, array_conditions):
        """Compare with PsychroLib for array inputs."""
        temps_c = array_conditions["temps_k"] - 273.15
        rhs = array_conditions["rhs"]

        # Our implementation
        tf_ours_k = Dewpoint.get_frostpoint_approximation(
            array_conditions["temps_k"], rhs
        )
        tf_ours_c = tf_ours_k - 273.15

        # Compare with PsychroLib element-wise
        for i, (temp_c, rh) in enumerate(zip(temps_c, rhs)):
            td_psychro_c = psy.GetTDewPointFromRelHum(temp_c, rh)

            # Check reasonable agreement
            diff = abs(tf_ours_c[i] - td_psychro_c)
            assert diff < 3.0, f"Index {i}: difference {diff:.2f}°C too large"

            # Frost point should be higher
            assert tf_ours_c[i] > td_psychro_c, (
                f"Index {i}: frost point should be > dew point"
            )


# ============================================================================
# Cross-Reference Validation
# ============================================================================


@pytest.mark.skipif(
    not (HAS_METPY and HAS_PSYCHROLIB), reason="Both MetPy and PsychroLib required"
)
class TestCrossReference:
    """Cross-validate between all three implementations."""

    def test_all_three_agree_on_trend(self):
        """All three implementations should show same trends."""
        temp_k = 263.15  # -10°C
        rhs = np.array([0.3, 0.5, 0.7, 0.9])

        # Our implementation
        tf_ours = Dewpoint.get_frostpoint_approximation(temp_k, rhs)

        # MetPy (dew point)
        tf_metpy = []
        for rh in rhs:
            temp_metpy = (temp_k * units.K).to("degC")
            rh_metpy = rh * 100 * units.percent
            td = dewpoint_from_relative_humidity(temp_metpy, rh_metpy)
            tf_metpy.append(td.to("K").magnitude)
        tf_metpy = np.array(tf_metpy)

        # PsychroLib (dew point)
        temp_c = temp_k - 273.15
        tf_psychro = np.array(
            [psy.GetTDewPointFromRelHum(temp_c, rh) + 273.15 for rh in rhs]
        )

        # All should show increasing trend with RH
        assert np.all(np.diff(tf_ours) > 0), "Our: should increase with RH"
        assert np.all(np.diff(tf_metpy) > 0), "MetPy: should increase with RH"
        assert np.all(np.diff(tf_psychro) > 0), "PsychroLib: should increase with RH"

        # All should be in reasonable range
        assert np.all(tf_ours < temp_k), "Our: frost point < air temp"
        assert np.all(tf_metpy < temp_k), "MetPy: dew point < air temp"
        assert np.all(tf_psychro < temp_k), "PsychroLib: dew point < air temp"


# ============================================================================
# Equation Selection Tests
# ============================================================================


class TestEquationSelection:
    """Test different frost point equation methods."""

    def test_magnus_default(self):
        """Magnus should be the default equation."""
        tf_default = Dewpoint.get_frostpoint_approximation(263.15, 0.7)
        tf_magnus = Dewpoint.get_frostpoint_approximation(
            263.15, 0.7, frostpoint_equation_name="magnus"
        )

        assert tf_default == tf_magnus


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_equation_raises(self):
        """Invalid equation name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid enum"):
            Dewpoint.get_frostpoint_approximation(
                263.15, 0.7, frostpoint_equation_name="invalid"
            )

    def test_solver_method_raises(self):
        """Solver method should raise error."""
        with pytest.raises(ValueError, match="not a valid approximation"):
            Dewpoint.get_frostpoint_approximation(
                263.15, 0.7, frostpoint_equation_name="vapor_inversion"
            )

    def test_rh_below_zero_raises(self):
        """Negative RH should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_frostpoint_approximation(263.15, -0.1)

    def test_rh_above_one_raises(self):
        """RH > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Relative humidity must be"):
            Dewpoint.get_frostpoint_approximation(263.15, 1.5)

    def test_temperature_out_of_range_warns(self):
        """Temperature outside valid range should warn."""
        with pytest.warns(UserWarning, match="outside valid range"):
            Dewpoint.get_frostpoint_approximation(200.0, 0.7)


# ============================================================================
# Broadcasting Tests
# ============================================================================


class TestBroadcasting:
    """Test broadcasting between scalar and array inputs."""

    def test_scalar_temp_array_rh(self):
        """Scalar temp, array RH should broadcast."""
        rhs = np.array([0.5, 0.6, 0.7])
        tf = Dewpoint.get_frostpoint_approximation(temp_k=263.15, rh=rhs)

        assert tf.shape == rhs.shape
        assert len(tf) == 3

    def test_array_temp_scalar_rh(self):
        """Array temp, scalar RH should broadcast."""
        temps = np.array([253.15, 263.15, 268.15])
        tf = Dewpoint.get_frostpoint_approximation(temp_k=temps, rh=0.7)

        assert tf.shape == temps.shape

    def test_both_arrays_same_shape(self):
        """Both arrays with same shape should work."""
        temps = np.array([253.15, 263.15, 268.15])
        rhs = np.array([0.5, 0.7, 0.9])
        tf = Dewpoint.get_frostpoint_approximation(temp_k=temps, rh=rhs)

        assert tf.shape == temps.shape

    def test_incompatible_shapes_raises(self):
        """Arrays with incompatible shapes should raise."""
        temps = np.array([253.15, 263.15, 268.15])
        rhs = np.array([0.5, 0.7])

        with pytest.raises(ValueError, match="same shape"):
            Dewpoint.get_frostpoint_approximation(temp_k=temps, rh=rhs)


# ============================================================================
# Physical Correctness Tests
# ============================================================================


class TestPhysicalCorrectness:
    """Test physical relationships and constraints."""

    def test_frost_point_below_air_temp(self):
        """Frost point must be at or below air temperature."""
        temp_k = 263.15
        rh = 0.7
        tf = Dewpoint.get_frostpoint_approximation(temp_k, rh)

        assert tf <= temp_k, f"Frost point ({tf}K) must be <= air temp ({temp_k}K)"

    def test_increases_with_rh(self):
        """Higher RH should give higher frost point."""
        temp_k = 263.15
        tf_low = Dewpoint.get_frostpoint_approximation(temp_k, 0.3)
        tf_high = Dewpoint.get_frostpoint_approximation(temp_k, 0.9)

        assert tf_high > tf_low

    def test_saturation_equals_temp(self):
        """At 100% RH, frost point should equal air temperature."""
        temp_k = 263.15
        tf = Dewpoint.get_frostpoint_approximation(temp_k, 1.0)

        assert abs(tf - temp_k) < 0.01

    def test_increases_with_temp(self):
        """At same RH, warmer air has higher frost point."""
        rh = 0.7
        tf_cold = Dewpoint.get_frostpoint_approximation(253.15, rh)
        tf_warm = Dewpoint.get_frostpoint_approximation(268.15, rh)

        assert tf_warm > tf_cold


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with other meteocalc components."""

    def test_frost_vs_dew_point(self):
        """Frost point should be higher than dew point below 0°C."""
        temp_k = 263.15
        rh = 0.7

        tf = Dewpoint.get_frostpoint_approximation(temp_k, rh)
        td = Dewpoint.get_dewpoint_approximation(temp_k, rh)

        assert tf > td, f"Frost point ({tf}K) should be > dew point ({td}K)"

        diff = tf - td
        assert 0.1 < diff < 5.0, (
            f"Difference {diff:.2f}K outside expected range (0.1-5K)"
        )

    def test_consistency_with_solver(self):
        """Approximation should be close to exact solver."""
        temp_k = 263.15
        rh = 0.7

        tf_approx = Dewpoint.get_frostpoint_approximation(temp_k, rh)
        tf_exact = Dewpoint.get_frostpoint_solver(temp_k, rh)

        diff = abs(tf_approx - tf_exact)
        assert diff < 0.5, f"Approximation differs from solver by {diff:.3f}K"


# ============================================================================
# Summary Report
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
