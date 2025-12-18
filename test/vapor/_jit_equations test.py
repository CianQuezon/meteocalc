"""
Unit tests for saturation vapor pressure calculations.

Tests validate:
- Scalar and vectorized implementations
- Cross-validation with psychrolib, MetPy, and NOAA values
- Edge cases and numerical stability
- Internal consistency

Author: Test Suite for Cian Quezon's meteorological equations
"""

import numpy as np
import pytest

from meteorological_equations.vapor._jit_equations import (
    _bolton_scalar,
    _bolton_vectorised,
    _goff_gratch_scalar,
    _goff_gratch_vector,
    _hyland_wexler_scalar,
    _hyland_wexler_vectorised,
)
from meteorological_equations.vapor._vapor_constants import (
    GOFF_GRATCH_WATER,
    GOFF_GRATCH_ICE,
    HYLAND_WEXLER_WATER,
    HYLAND_WEXLER_ICE,
)


# ============================================================================
# Reference Values from NOAA Standard Atmosphere Tables
# ============================================================================

NOAA_REFERENCE_VALUES = [
    # (temp_k, expected_es_hPa, source)
    (273.15, 6.11, "NOAA - 0°C"),
    (283.15, 12.28, "NOAA - 10°C"),
    (293.15, 23.39, "NOAA - 20°C"),
    (298.15, 31.69, "NOAA - 25°C"),
    (303.15, 42.44, "NOAA - 30°C"),
    (313.15, 73.77, "NOAA - 40°C"),
]


# ============================================================================
# Bolton Equation Tests
# ============================================================================

class TestBoltonEquation:
    """Test suite for Bolton's equation."""
    
    @pytest.mark.parametrize("temp_k,expected_es,source", NOAA_REFERENCE_VALUES)
    def test_scalar_against_noaa(self, temp_k: float, expected_es: float, source: str):
        """Validate Bolton scalar implementation against NOAA reference values."""
        result = _bolton_scalar(temp_k)
        tolerance = expected_es * 0.05  # 5% tolerance for Bolton approximation
        assert abs(result - expected_es) < tolerance, \
            f"Failed for {source}: {result:.2f} hPa vs {expected_es:.2f} hPa"
    
    def test_scalar_at_freezing_point(self):
        """Test Bolton at water freezing point."""
        result = _bolton_scalar(273.15)
        assert 6.0 < result < 6.2, f"Expected ~6.11 hPa, got {result:.2f} hPa"
    
    def test_vectorised_matches_scalar(self):
        """Verify vectorized implementation matches scalar."""
        temps = np.array([270.0, 280.0, 290.0, 300.0, 310.0])
        vec_result = _bolton_vectorised(temps)
        
        for i, temp in enumerate(temps):
            scalar_result = _bolton_scalar(temp)
            assert abs(vec_result[i] - scalar_result) < 1e-10, \
                f"Mismatch at {temp}K: {vec_result[i]} vs {scalar_result}"
    
    def test_vectorised_output_shape(self):
        """Test vectorized output has correct shape and dtype."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        result = _bolton_vectorised(temps)
        
        assert result.shape == temps.shape
        assert result.dtype == np.float64
        assert np.all(result > 0)
    
    def test_monotonic_increase(self):
        """Verify vapor pressure increases monotonically with temperature."""
        temps = np.linspace(233.15, 323.15, 50)
        results = _bolton_vectorised(temps)
        
        assert np.all(np.diff(results) > 0), "Vapor pressure should increase monotonically"


# ============================================================================
# Goff-Gratch Equation Tests
# ============================================================================

class TestGoffGratchEquation:
    """Test suite for Goff-Gratch equation."""
    
    @pytest.mark.parametrize("temp_k,expected_es,source", NOAA_REFERENCE_VALUES)
    def test_water_against_noaa(self, temp_k: float, expected_es: float, source: str):
        """Validate Goff-Gratch water equation against NOAA values."""
        result = _goff_gratch_scalar(temp_k, *GOFF_GRATCH_WATER)
        tolerance = expected_es * 0.02  # 2% tolerance for WMO standard
        assert abs(result - expected_es) < tolerance, \
            f"Failed for {source}: {result:.2f} hPa vs {expected_es:.2f} hPa"
    
    def test_at_reference_temperature(self):
        """Test Goff-Gratch returns reference pressure at reference temperature."""
        result = _goff_gratch_scalar(
            GOFF_GRATCH_WATER.T_ref, *GOFF_GRATCH_WATER
        )
        expected = 1013.246
        tolerance = expected * 0.001  # 0.1% tolerance
        assert abs(result - expected) < tolerance, \
            f"Expected {expected:.2f} hPa, got {result:.2f} hPa"
    
    def test_ice_at_triple_point(self):
        """Test Goff-Gratch ice equation at triple point."""
        result = _goff_gratch_scalar(273.16, *GOFF_GRATCH_ICE)
        expected = 6.1071
        assert abs(result - expected) < 0.01, \
            f"Expected {expected:.2f} hPa, got {result:.2f} hPa"
    
    def test_vectorised_matches_scalar(self):
        """Verify vectorized Goff-Gratch matches scalar."""
        temps = np.array([273.15, 283.15, 293.15, 303.15, 313.15])
        vec_result = _goff_gratch_vector(temps, *GOFF_GRATCH_WATER)
        
        for i, temp in enumerate(temps):
            scalar_result = _goff_gratch_scalar(temp, *GOFF_GRATCH_WATER)
            assert abs(vec_result[i] - scalar_result) < 1e-10, \
                f"Mismatch at {temp}K"
    
    def test_monotonic_increase(self):
        """Verify vapor pressure increases with temperature."""
        temps = np.array([280.0, 290.0, 300.0, 310.0, 320.0])
        results = _goff_gratch_vector(temps, *GOFF_GRATCH_WATER)
        
        assert np.all(np.diff(results) > 0), "Should increase monotonically"


# ============================================================================
# Hyland-Wexler Equation Tests
# ============================================================================

class TestHylandWexlerEquation:
    """Test suite for Hyland-Wexler equation."""
    
    @pytest.mark.parametrize("temp_k,expected_es,source", NOAA_REFERENCE_VALUES)
    def test_water_against_noaa(self, temp_k: float, expected_es: float, source: str):
        """Validate Hyland-Wexler water equation against NOAA values."""
        result = _hyland_wexler_scalar(temp_k, *HYLAND_WEXLER_WATER)
        tolerance = expected_es * 0.02  # 2% tolerance for ASHRAE standard
        assert abs(result - expected_es) < tolerance, \
            f"Failed for {source}: {result:.2f} hPa vs {expected_es:.2f} hPa"
    
    def test_vectorised_matches_scalar(self):
        """Verify vectorized Hyland-Wexler matches scalar."""
        temps = np.array([273.15, 283.15, 293.15, 303.15, 313.15])
        vec_result = _hyland_wexler_vectorised(temps, *HYLAND_WEXLER_WATER)
        
        for i, temp in enumerate(temps):
            scalar_result = _hyland_wexler_scalar(temp, *HYLAND_WEXLER_WATER)
            assert abs(vec_result[i] - scalar_result) < 1e-10, \
                f"Mismatch at {temp}K"
    
    def test_output_shape_and_dtype(self):
        """Test vectorized output properties."""
        temps = np.linspace(280, 360, 25)
        result = _hyland_wexler_vectorised(temps, *HYLAND_WEXLER_WATER)
        
        assert result.shape == temps.shape
        assert result.dtype == np.float64
        assert np.all(result > 0)
    
    def test_monotonic_increase(self):
        """Verify vapor pressure increases with temperature."""
        temps = np.linspace(280.0, 360.0, 50)
        results = _hyland_wexler_vectorised(temps, *HYLAND_WEXLER_WATER)
        
        assert np.all(np.diff(results) > 0), "Should increase monotonically"

# ============================================================================
# Cross-Validation with External Libraries
# ============================================================================

class TestCrossValidation:
    """Cross-validate against psychrolib and MetPy."""
    
    def test_bolton_vs_metpy(self):
        """Compare Bolton implementation with MetPy."""
        try:
            from metpy.calc import saturation_vapor_pressure
            from metpy.units import units
            
            temps_c = np.array([0, 10, 20, 25, 30])
            temps_k = temps_c + 273.15
            
            bolton_results = _bolton_vectorised(temps_k)
            metpy_results = saturation_vapor_pressure(temps_c * units.degC).to('hPa').magnitude
            
            # MetPy uses same Bolton formula, should match within 0.5%
            rel_diff = np.abs(bolton_results - metpy_results) / metpy_results
            assert np.all(rel_diff < 0.005), \
                f"Bolton vs MetPy difference >0.5%: max={np.max(rel_diff)*100:.2f}%"
            
        except ImportError:
            pytest.skip("MetPy not installed")
    
    def test_bolton_vs_psychrolib(self):
        """Compare Bolton implementation with psychrolib."""
        try:
            import psychrolib
            psychrolib.SetUnitSystem(psychrolib.SI)
            
            temps_c = np.array([0, 10, 20, 25, 30])
            temps_k = temps_c + 273.15
            
            bolton_results = _bolton_vectorised(temps_k)
            psychro_results = np.array([
                psychrolib.GetSatVapPres(t) / 100  # Convert Pa to hPa
                for t in temps_c
            ])
            
            # Different formulations, allow 5% difference
            rel_diff = np.abs(bolton_results - psychro_results) / psychro_results
            assert np.all(rel_diff < 0.05), \
                f"Bolton vs psychrolib difference >5%: max={np.max(rel_diff)*100:.1f}%"
            
        except ImportError:
            pytest.skip("psychrolib not installed")
    
    def test_goff_gratch_vs_psychrolib(self):
        """Compare Goff-Gratch implementation with psychrolib."""
        try:
            import psychrolib
            psychrolib.SetUnitSystem(psychrolib.SI)
            
            temps_c = np.array([0, 10, 20, 25, 30])
            temps_k = temps_c + 273.15
            
            goff_gratch_results = _goff_gratch_vector(temps_k, *GOFF_GRATCH_WATER)
            psychro_results = np.array([
                psychrolib.GetSatVapPres(t) / 100  # Convert Pa to hPa
                for t in temps_c
            ])
            
            # Both are standard formulations, should be within 3%
            rel_diff = np.abs(goff_gratch_results - psychro_results) / psychro_results
            assert np.all(rel_diff < 0.03), \
                f"Goff-Gratch vs psychrolib difference >3%: max={np.max(rel_diff)*100:.1f}%"
            
        except ImportError:
            pytest.skip("psychrolib not installed")
    
    def test_hyland_wexler_vs_psychrolib(self):
        """Compare Hyland-Wexler implementation with psychrolib."""
        try:
            import psychrolib
            psychrolib.SetUnitSystem(psychrolib.SI)
            
            temps_c = np.array([0, 10, 20, 25, 30])
            temps_k = temps_c + 273.15
            
            hyland_wexler_results = _hyland_wexler_vectorised(temps_k, *HYLAND_WEXLER_WATER)
            psychro_results = np.array([
                psychrolib.GetSatVapPres(t) / 100  # Convert Pa to hPa
                for t in temps_c
            ])
            
            # Different formulations, allow 5% difference
            rel_diff = np.abs(hyland_wexler_results - psychro_results) / psychro_results
            assert np.all(rel_diff < 0.05), \
                f"Hyland-Wexler vs psychrolib difference >5%: max={np.max(rel_diff)*100:.1f}%"
            
        except ImportError:
            pytest.skip("psychrolib not installed")
    
    def test_all_equations_consistency(self):
        """
        Verify all three equations give reasonably consistent results.
        
        While different formulations, they should all be within 10% of each other
        for standard meteorological temperatures.
        """
        temps_k = np.array([273.15, 283.15, 293.15, 303.15, 313.15])
        
        bolton = _bolton_vectorised(temps_k)
        goff_gratch = _goff_gratch_vector(temps_k, *GOFF_GRATCH_WATER)
        hyland_wexler = _hyland_wexler_vectorised(temps_k, *HYLAND_WEXLER_WATER)
        
        # Compare each pair
        for i, temp in enumerate(temps_k):
            values = np.array([bolton[i], goff_gratch[i], hyland_wexler[i]])
            mean_val = np.mean(values)
            rel_diffs = np.abs(values - mean_val) / mean_val
            
            assert np.all(rel_diffs < 0.1), \
                f"Equations disagree by >10% at {temp:.2f}K: " \
                f"Bolton={bolton[i]:.2f}, GG={goff_gratch[i]:.2f}, HW={hyland_wexler[i]:.2f}"
    
    def test_goff_gratch_vs_metpy(self):
        """Compare Goff-Gratch with MetPy (if available)."""
        try:
            from metpy.calc import saturation_vapor_pressure
            from metpy.units import units
            
            temps_c = np.array([0, 10, 20, 25, 30])
            temps_k = temps_c + 273.15
            
            goff_gratch_results = _goff_gratch_vector(temps_k, *GOFF_GRATCH_WATER)
            metpy_results = saturation_vapor_pressure(temps_c * units.degC).to('hPa').magnitude
            
            # MetPy uses Bolton, Goff-Gratch is different - allow 10% difference
            rel_diff = np.abs(goff_gratch_results - metpy_results) / metpy_results
            assert np.all(rel_diff < 0.10), \
                f"Goff-Gratch vs MetPy difference >10%: max={np.max(rel_diff)*100:.1f}%"
            
        except ImportError:
            pytest.skip("MetPy not installed")
    
    def test_hyland_wexler_vs_metpy(self):
        """Compare Hyland-Wexler with MetPy (if available)."""
        try:
            from metpy.calc import saturation_vapor_pressure
            from metpy.units import units
            
            temps_c = np.array([0, 10, 20, 25, 30])
            temps_k = temps_c + 273.15
            
            hyland_wexler_results = _hyland_wexler_vectorised(temps_k, *HYLAND_WEXLER_WATER)
            metpy_results = saturation_vapor_pressure(temps_c * units.degC).to('hPa').magnitude
            
            # MetPy uses Bolton, Hyland-Wexler is different - allow 10% difference
            rel_diff = np.abs(hyland_wexler_results - metpy_results) / metpy_results
            assert np.all(rel_diff < 0.10), \
                f"Hyland-Wexler vs MetPy difference >10%: max={np.max(rel_diff)*100:.1f}%"
            
        except ImportError:
            pytest.skip("MetPy not installed")

# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_bolton_extreme_temperatures(self):
        """Test Bolton at temperature extremes."""
        cold = _bolton_scalar(233.15)  # -40°C
        hot = _bolton_scalar(323.15)   # 50°C
        
        assert 0 < cold < 2, "Vapor pressure should be small at -40°C"
        assert 100 < hot < 150, "Vapor pressure should be high at 50°C"
    
    def test_empty_array_handling(self):
        """Test vectorized functions with empty arrays."""
        empty = np.array([], dtype=np.float64)
        
        result_bolton = _bolton_vectorised(empty)
        result_gg = _goff_gratch_vector(empty, *GOFF_GRATCH_WATER)
        result_hw = _hyland_wexler_vectorised(empty, *HYLAND_WEXLER_WATER)
        
        assert len(result_bolton) == 0
        assert len(result_gg) == 0
        assert len(result_hw) == 0
    
    def test_single_element_array(self):
        """Test vectorized functions with single element."""
        single = np.array([293.15])
        
        vec_bolton = _bolton_vectorised(single)[0]
        scalar_bolton = _bolton_scalar(293.15)
        
        assert abs(vec_bolton - scalar_bolton) < 1e-10
    
    def test_no_nan_or_inf_values(self):
        """Ensure no NaN or Inf values in valid range."""
        temps = np.linspace(233.15, 373.15, 100)
        
        bolton_results = _bolton_vectorised(temps)
        gg_results = _goff_gratch_vector(temps, *GOFF_GRATCH_WATER)
        hw_results = _hyland_wexler_vectorised(temps, *HYLAND_WEXLER_WATER)
        
        assert not np.any(np.isnan(bolton_results))
        assert not np.any(np.isinf(bolton_results))
        assert not np.any(np.isnan(gg_results))
        assert not np.any(np.isinf(gg_results))
        assert not np.any(np.isnan(hw_results))
        assert not np.any(np.isinf(hw_results))


# ============================================================================
# Numerical Precision Tests
# ============================================================================

class TestNumericalPrecision:
    """Test numerical precision and consistency."""
    
    def test_small_temperature_changes(self):
        """Test sensitivity to small temperature changes."""
        base_temp = 293.15
        delta = 0.01  # 0.01K difference
        
        result1 = _bolton_scalar(base_temp)
        result2 = _bolton_scalar(base_temp + delta)
        
        assert result1 != result2, "Should detect small temperature differences"
        rel_diff = abs(result2 - result1) / result1
        assert rel_diff < 0.01, "Small temp change should cause small pressure change"
    
    def test_equation_consistency(self):
        """Compare all three equations at same temperature."""
        temp = 298.15  # 25°C
        
        bolton = _bolton_scalar(temp)
        goff_gratch = _goff_gratch_scalar(temp, *GOFF_GRATCH_WATER)
        hyland_wexler = _hyland_wexler_scalar(temp, *HYLAND_WEXLER_WATER)
        
        # All should be within 10% of each other (different formulations)
        values = np.array([bolton, goff_gratch, hyland_wexler])
        mean_val = np.mean(values)
        rel_diffs = np.abs(values - mean_val) / mean_val
        
        assert np.all(rel_diffs < 0.1), \
            f"Equations disagree by >10%: {bolton:.2f}, {goff_gratch:.2f}, {hyland_wexler:.2f}"


# ============================================================================
# Constants Validation
# ============================================================================

class TestConstants:
    """Validate constant definitions."""
    
    def test_goff_gratch_water_constants(self):
        """Verify Goff-Gratch water constants are reasonable."""
        assert GOFF_GRATCH_WATER.T_ref > 0
        assert GOFF_GRATCH_WATER.log_p_ref > 0
        assert np.isclose(GOFF_GRATCH_WATER.log_p_ref, np.log10(1013.246))
    
    def test_goff_gratch_ice_constants(self):
        """Verify Goff-Gratch ice constants are reasonable."""
        assert GOFF_GRATCH_ICE.T_ref > 0
        assert GOFF_GRATCH_ICE.log_p_ref > 0
        assert GOFF_GRATCH_ICE.D == 0.0  # Ice equation omits D term
        assert GOFF_GRATCH_ICE.D_exp == 0.0
    
    def test_hyland_wexler_water_constants(self):
        """Verify Hyland-Wexler water constants."""
        assert HYLAND_WEXLER_WATER.F == 0.0  # Water equation omits T⁴ term
        assert HYLAND_WEXLER_WATER.A < 0  # Negative A coefficient


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])