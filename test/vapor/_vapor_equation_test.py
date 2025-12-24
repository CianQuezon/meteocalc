"""
Unit tests for vapor pressure equations.

Cross-validates calculations against:
- MetPy
- PsychroLib  
- NOAA reference values

Author: Test Suite
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

# Import the equations to test
from meteorological_equations.vapor._vapor_equations import (
    BoltonEquation,
    GoffGratchEquation,
    HylandWexlerEquation,
    VaporEquation
)
from meteorological_equations.vapor._enums import SurfaceType

# Import reference libraries
try:
    import metpy.calc as mpcalc
    from metpy.units import units
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False

try:
    import psychrolib as psy
    # Set unit system to SI
    psy.SetUnitSystem(psy.SI)
    PSYCHROLIB_AVAILABLE = True
except ImportError:
    PSYCHROLIB_AVAILABLE = False


class TestBoltonEquation:
    """Test Bolton equation against reference values."""
    
    @pytest.fixture
    def bolton(self):
        return BoltonEquation()
    
    def test_bolton_only_supports_water(self):
        """Bolton equation should raise error for ice or automatic."""
        with pytest.raises(ValueError, match="Bolton only supports water"):
            BoltonEquation(surface_type=SurfaceType.ICE)
        
        with pytest.raises(ValueError, match="Bolton only supports water"):
            BoltonEquation(surface_type=SurfaceType.AUTOMATIC)
    
    def test_bolton_scalar_calculation(self, bolton):
        """Test Bolton equation with scalar input against MetPy reference."""
        # Reference: MetPy saturation_vapor_pressure at 20°C (293.15K)
        # Expected: ~23.37 hPa
        temp_k = 293.15
        result = bolton.calculate(temp_k)
        
        assert isinstance(result, (float, np.floating))
        assert_allclose(result, 23.37, rtol=0.02)  # 2% tolerance
    
    def test_bolton_vector_calculation(self, bolton):
        """Test Bolton equation with array input."""
        temps_k = np.array([273.15, 283.15, 293.15, 303.15])
        # Expected values from MetPy (approximate)
        expected = np.array([6.11, 12.27, 23.37, 42.43])
        
        result = bolton.calculate(temps_k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == temps_k.shape
        assert_allclose(result, expected, rtol=0.02)
    
    def test_bolton_temperature_bounds_warning(self, bolton):
        """Test that Bolton warns for out-of-bounds temperatures."""
        with pytest.warns(UserWarning, match="outside the valid range"):
            bolton.calculate(200.0)  # Below 243.15K
        
        with pytest.warns(UserWarning, match="outside the valid range"):
            bolton.calculate(350.0)  # Above 313.15K


class TestGoffGratchEquation:
    """Test Goff-Gratch equation against NOAA and reference values."""
    
    @pytest.fixture
    def goff_gratch_water(self):
        return GoffGratchEquation(surface_type=SurfaceType.WATER)
    
    @pytest.fixture
    def goff_gratch_ice(self):
        return GoffGratchEquation(surface_type=SurfaceType.ICE)
    
    @pytest.fixture
    def goff_gratch_auto(self):
        return GoffGratchEquation(surface_type=SurfaceType.AUTOMATIC)
    
    def test_goff_gratch_water_scalar(self, goff_gratch_water):
        """Test Goff-Gratch over water at 25°C against NOAA tables."""
        # NOAA reference: 25°C (298.15K) ≈ 31.67 hPa
        temp_k = 298.15
        result = goff_gratch_water.calculate(temp_k)
        
        assert isinstance(result, (float, np.floating))
        assert_allclose(result, 31.67, rtol=0.01)
    
    def test_goff_gratch_ice_scalar(self, goff_gratch_ice):
        """Test Goff-Gratch over ice at -10°C."""
        # Reference: -10°C (263.15K) ≈ 2.60 hPa
        temp_k = 263.15
        result = goff_gratch_ice.calculate(temp_k)
        
        assert isinstance(result, (float, np.floating))
        assert_allclose(result, 2.60, rtol=0.02)
    
    def test_goff_gratch_automatic_mixed_temps(self, goff_gratch_auto):
        """Test automatic surface detection with mixed temperatures."""
        temps_k = np.array([263.15, 273.15, 283.15])  # Below, at, above freezing
        
        result = goff_gratch_auto.calculate(temps_k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        # Ice: ~2.60, Freezing: avg, Water: ~12.27
        assert result[0] < result[1] < result[2]
    
    def test_goff_gratch_freezing_point_average(self, goff_gratch_auto):
        """Test that at freezing point, automatic mode averages ice and water."""
        temp_k = 273.15
        
        goff_gratch_water = GoffGratchEquation(surface_type=SurfaceType.WATER)
        goff_gratch_ice = GoffGratchEquation(surface_type=SurfaceType.ICE)
        
        result_auto = goff_gratch_auto.calculate(temp_k)
        result_water = goff_gratch_water.calculate(temp_k)
        result_ice = goff_gratch_ice.calculate(temp_k)
        
        expected_avg = (result_water + result_ice) / 2
        assert_allclose(result_auto, expected_avg, rtol=1e-10)
    
    def test_goff_gratch_temp_bounds(self):
        """Test temperature bounds are set correctly."""
        water_eq = GoffGratchEquation(surface_type=SurfaceType.WATER)
        ice_eq = GoffGratchEquation(surface_type=SurfaceType.ICE)
        auto_eq = GoffGratchEquation(surface_type=SurfaceType.AUTOMATIC)
        
        assert water_eq.temp_bounds == (273.15, 373.15)
        assert ice_eq.temp_bounds == (173.15, 273.16)
        assert auto_eq.temp_bounds == (173.15, 373.15)


class TestHylandWexlerEquation:
    """Test Hyland-Wexler equation against PsychroLib and reference values."""
    
    @pytest.fixture
    def hyland_wexler_water(self):
        return HylandWexlerEquation(surface_type=SurfaceType.WATER)
    
    @pytest.fixture
    def hyland_wexler_ice(self):
        return HylandWexlerEquation(surface_type=SurfaceType.ICE)
    
    @pytest.fixture
    def hyland_wexler_auto(self):
        return HylandWexlerEquation(surface_type=SurfaceType.AUTOMATIC)
    
    def test_hyland_wexler_water_reference(self, hyland_wexler_water):
        """Test Hyland-Wexler over water at 20°C (PsychroLib reference)."""
        # PsychroLib reference: 20°C ≈ 23.39 hPa
        temp_k = 293.15
        result = hyland_wexler_water.calculate(temp_k)
        
        assert isinstance(result, (float, np.floating))
        assert_allclose(result, 23.39, rtol=0.01)
    
    def test_hyland_wexler_ice_reference(self, hyland_wexler_ice):
        """Test Hyland-Wexler over ice at -20°C."""
        # Reference: -20°C (253.15K) ≈ 1.03 hPa
        temp_k = 253.15
        result = hyland_wexler_ice.calculate(temp_k)
        
        assert isinstance(result, (float, np.floating))
        assert_allclose(result, 1.03, rtol=0.02)
    
    def test_hyland_wexler_vector_shape_preservation(self, hyland_wexler_water):
        """Test that output shape matches input shape."""
        temps_2d = np.array([[273.15, 283.15], [293.15, 303.15]])
        result = hyland_wexler_water.calculate(temps_2d)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == temps_2d.shape
    
    def test_hyland_wexler_automatic_transition(self, hyland_wexler_auto):
        """Test smooth transition around freezing point."""
        temps_k = np.linspace(271.15, 275.15, 9)  # Around freezing
        result = hyland_wexler_auto.calculate(temps_k)
        
        assert isinstance(result, np.ndarray)
        assert np.all(np.diff(result) > 0)  # Should be monotonically increasing
    
    def test_hyland_wexler_temp_bounds(self):
        """Test temperature bounds are set correctly."""
        water_eq = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        ice_eq = HylandWexlerEquation(surface_type=SurfaceType.ICE)
        auto_eq = HylandWexlerEquation(surface_type=SurfaceType.AUTOMATIC)
        
        assert water_eq.temp_bounds == (273.15, 473.15)
        assert ice_eq.temp_bounds == (173.15, 273.16)
        assert auto_eq.temp_bounds == (173.15, 473.15)


class TestVaporEquationBase:
    """Test base class functionality."""
    
    def test_surface_type_string_conversion(self):
        """Test that string surface types are converted to enum."""
        eq = GoffGratchEquation(surface_type="water")
        assert eq.surface_type == SurfaceType.WATER
        
        eq = GoffGratchEquation(surface_type="ice")
        assert eq.surface_type == SurfaceType.ICE
    
    def test_invalid_surface_type_string(self):
        """Test that invalid surface type strings raise error."""
        with pytest.raises(ValueError, match="Invalid surface type"):
            GoffGratchEquation(surface_type="invalid")
    
    def test_detect_surface_type_scalar(self):
        """Test surface type detection for scalar temperatures."""
        eq = GoffGratchEquation(surface_type=SurfaceType.AUTOMATIC)
        
        # Above freezing
        at, above, below = eq._detect_surface_type(280.0)
        assert above.item() and not at.item() and not below.item()
        
        # Below freezing
        at, above, below = eq._detect_surface_type(265.0)
        assert below.item() and not at.item() and not above.item()
        
        # At freezing
        at, above, below = eq._detect_surface_type(273.15)
        assert at.item() and not above.item() and not below.item()
    
    def test_detect_surface_type_array(self):
        """Test surface type detection for array temperatures."""
        eq = GoffGratchEquation(surface_type=SurfaceType.AUTOMATIC)
        temps = np.array([265.0, 273.15, 280.0])
        
        at, above, below = eq._detect_surface_type(temps)
        
        assert below[0] and at[1] and above[2]


class TestCrossValidation:
    """Cross-validation tests comparing different equations."""
    
    def test_bolton_vs_hyland_wexler_water(self):
        """Bolton and Hyland-Wexler should give similar results for water."""
        bolton = BoltonEquation()
        hyland = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        
        # Test in Bolton's valid range
        temps_k = np.array([250.0, 270.0, 290.0, 310.0])
        
        bolton_result = bolton.calculate(temps_k)
        hyland_result = hyland.calculate(temps_k)
        
        # Should be within 5% of each other (different formulations)
        assert_allclose(bolton_result, hyland_result, rtol=0.05)
    
    def test_triple_point_consistency(self):
        """Test that all equations give reasonable values at triple point."""
        # Triple point of water: 273.16K, ~6.11 hPa
        temp_k = 273.16
        
        bolton = BoltonEquation()
        goff_gratch = GoffGratchEquation(surface_type=SurfaceType.WATER)
        hyland = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        
        bolton_result = bolton.calculate(temp_k)
        goff_result = goff_gratch.calculate(temp_k)
        hyland_result = hyland.calculate(temp_k)
        
        # All should be close to 6.11 hPa
        assert_allclose([bolton_result, goff_result, hyland_result], 6.11, rtol=0.03)


class TestMetPyValidation:
    """Validate against MetPy library values."""
    
    @pytest.mark.skipif(not METPY_AVAILABLE, reason="MetPy not installed")
    def test_bolton_vs_metpy_scalar(self):
        """Validate Bolton equation against MetPy for scalar temperatures."""
        bolton = BoltonEquation()
        
        test_temps_c = [0, 10, 20, 30]
        
        for temp_c in test_temps_c:
            temp_k = temp_c + 273.15
            
            # Get MetPy result
            metpy_result = mpcalc.saturation_vapor_pressure(temp_c * units.degC)
            metpy_hpa = metpy_result.to('hPa').magnitude
            
            # Get our result
            our_result = bolton.calculate(temp_k)
            
            assert_allclose(our_result, metpy_hpa, rtol=0.015,
                           err_msg=f"Failed at {temp_c}°C")
    
    @pytest.mark.skipif(not METPY_AVAILABLE, reason="MetPy not installed")
    def test_bolton_vs_metpy_array(self):
        """Validate Bolton equation against MetPy for array temperatures."""
        bolton = BoltonEquation()
        
        temps_c = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
        temps_k = temps_c + 273.15
        
        # Get MetPy results
        metpy_results = mpcalc.saturation_vapor_pressure(temps_c * units.degC)
        metpy_hpa = metpy_results.to('hPa').magnitude
        
        # Get our results
        our_results = bolton.calculate(temps_k)
        
        assert_allclose(our_results, metpy_hpa, rtol=0.015)
    
    @pytest.mark.skipif(not METPY_AVAILABLE, reason="MetPy not installed")
    def test_goff_gratch_water_vs_metpy(self):
        """Validate Goff-Gratch over water against MetPy."""
        goff = GoffGratchEquation(surface_type=SurfaceType.WATER)
        
        temps_c = np.array([0, 10, 20, 25, 30, 40, 50])
        temps_k = temps_c + 273.15
        
        # MetPy uses Goff-Gratch by default for water
        metpy_results = mpcalc.saturation_vapor_pressure(temps_c * units.degC)
        metpy_hpa = metpy_results.to('hPa').magnitude
        
        our_results = goff.calculate(temps_k)
        
        # Should match very closely since MetPy uses Goff-Gratch
        assert_allclose(our_results, metpy_hpa, rtol=0.005)
    
    @pytest.mark.skipif(not METPY_AVAILABLE, reason="MetPy not installed")
    def test_goff_gratch_ice_vs_metpy(self):
        """Validate Goff-Gratch over ice against MetPy."""
        goff = GoffGratchEquation(surface_type=SurfaceType.ICE)
        
        temps_c = np.array([-40, -30, -20, -10, -5])
        temps_k = temps_c + 273.15
        
        # MetPy ice saturation vapor pressure
        metpy_results = mpcalc.saturation_vapor_pressure(temps_c * units.degC)
        metpy_hpa = metpy_results.to('hPa').magnitude
        
        our_results = goff.calculate(temps_k)
        
        assert_allclose(our_results, metpy_hpa, rtol=0.015)
    
    @pytest.mark.skipif(not METPY_AVAILABLE, reason="MetPy not installed")
    def test_hyland_wexler_vs_metpy(self):
        """Validate Hyland-Wexler equation against MetPy."""
        hyland = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        
        temps_c = np.array([0, 10, 20, 30, 40])
        temps_k = temps_c + 273.15
        
        metpy_results = mpcalc.saturation_vapor_pressure(temps_c * units.degC)
        metpy_hpa = metpy_results.to('hPa').magnitude
        
        our_results = hyland.calculate(temps_k)
        
        # Different formulations, but should be close
        assert_allclose(our_results, metpy_hpa, rtol=0.02)


class TestPsychroLibValidation:
    """Validate against PsychroLib library values."""
    
    @pytest.mark.skipif(not PSYCHROLIB_AVAILABLE, reason="PsychroLib not installed")
    def test_hyland_wexler_water_vs_psychrolib(self):
        """
        Validate Hyland-Wexler against PsychroLib (uses Hyland-Wexler formulation).
        
        PsychroLib returns pressure in Pa.
        """
        hyland = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        
        temps_c = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
        
        for temp_c in temps_c:
            temp_k = temp_c + 273.15
            
            # PsychroLib GetSatVapPres returns Pa
            psychro_pa = psy.GetSatVapPres(temp_c)
            psychro_hpa = psychro_pa / 100.0
            
            our_result = hyland.calculate(temp_k)
            
            # Should match very closely - both use Hyland-Wexler
            assert_allclose(our_result, psychro_hpa, rtol=0.005,
                           err_msg=f"Failed at {temp_c}°C")
    
    @pytest.mark.skipif(not PSYCHROLIB_AVAILABLE, reason="PsychroLib not installed")
    def test_hyland_wexler_ice_vs_psychrolib(self):
        """Validate Hyland-Wexler over ice against PsychroLib."""
        hyland = HylandWexlerEquation(surface_type=SurfaceType.ICE)
        
        temps_c = np.array([-30, -20, -10, -5])
        
        for temp_c in temps_c:
            temp_k = temp_c + 273.15
            
            # PsychroLib automatically uses ice formulation for T < 0°C
            psychro_pa = psy.GetSatVapPres(temp_c)
            psychro_hpa = psychro_pa / 100.0
            
            our_result = hyland.calculate(temp_k)
            
            assert_allclose(our_result, psychro_hpa, rtol=0.005,
                           err_msg=f"Failed at {temp_c}°C")
    
    @pytest.mark.skipif(not PSYCHROLIB_AVAILABLE, reason="PsychroLib not installed")
    def test_hyland_wexler_array_vs_psychrolib(self):
        """Test array calculations against PsychroLib."""
        hyland_water = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        
        temps_c = np.array([0, 10, 20, 25, 30])
        temps_k = temps_c + 273.15
        
        # Get PsychroLib results
        psychro_results = np.array([psy.GetSatVapPres(t) for t in temps_c])
        psychro_hpa = psychro_results / 100.0
        
        # Get our results
        our_results = hyland_water.calculate(temps_k)
        
        assert_allclose(our_results, psychro_hpa, rtol=0.005)


class TestNOAAValidation:
    """Validate against NOAA reference tables and standards."""
    
    @pytest.mark.parametrize("temp_c,expected_hpa,description", [
        # NOAA vapor pressure tables (converted from mmHg to hPa)
        (0, 6.11, "Freezing point - NOAA standard"),
        (5, 8.72, "NOAA table value"),
        (10, 12.27, "NOAA table value"),
        (15, 17.04, "NOAA table value"),
        (20, 23.37, "NOAA table value"),
        (25, 31.67, "NOAA table value"),
        (30, 42.43, "NOAA table value"),
        (35, 56.24, "NOAA table value"),
        (40, 73.78, "NOAA table value"),
    ])
    def test_goff_gratch_vs_noaa_tables(self, temp_c, expected_hpa, description):
        """
        Validate Goff-Gratch against NOAA vapor pressure tables.
        
        NOAA uses Goff-Gratch formulation as a standard.
        Reference: NOAA Technical Report NWS 23
        """
        goff = GoffGratchEquation(surface_type=SurfaceType.WATER)
        temp_k = temp_c + 273.15
        
        result = goff.calculate(temp_k)
        assert_allclose(result, expected_hpa, rtol=0.01, 
                       err_msg=f"Failed for {description}")
    
    @pytest.mark.parametrize("temp_c,expected_hpa", [
        # NOAA ice vapor pressure values
        (-40, 0.128),
        (-30, 0.380),
        (-20, 1.032),
        (-10, 2.599),
        (0, 6.11),
    ])
    def test_goff_gratch_ice_vs_noaa(self, temp_c, expected_hpa):
        """Validate Goff-Gratch over ice against NOAA tables."""
        goff = GoffGratchEquation(surface_type=SurfaceType.ICE)
        temp_k = temp_c + 273.15
        
        result = goff.calculate(temp_k)
        assert_allclose(result, expected_hpa, rtol=0.015)
    
    def test_standard_atmosphere_sea_level(self):
        """Test against NOAA standard atmosphere at sea level, 15°C."""
        # Standard atmosphere: 15°C (288.15K), vapor pressure ~17.04 hPa
        temp_k = 288.15
        
        goff = GoffGratchEquation(surface_type=SurfaceType.WATER)
        result = goff.calculate(temp_k)
        
        assert_allclose(result, 17.04, rtol=0.01)
    
    def test_wmo_triple_point(self):
        """
        Test WMO/NOAA triple point of water definition.
        
        Triple point: 273.16K, 6.1115 hPa (exact by definition)
        """
        temp_k = 273.16
        
        goff = GoffGratchEquation(surface_type=SurfaceType.WATER)
        hyland = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        
        goff_result = goff.calculate(temp_k)
        hyland_result = hyland.calculate(temp_k)
        
        # Should be very close to 6.1115 hPa
        assert_allclose(goff_result, 6.1115, rtol=0.005)
        assert_allclose(hyland_result, 6.1115, rtol=0.005)


class TestComprehensiveCrossValidation:
    """Comprehensive cross-validation across all reference sources."""
    
    @pytest.mark.skipif(not (METPY_AVAILABLE and PSYCHROLIB_AVAILABLE), 
                       reason="MetPy and PsychroLib required")
    def test_all_libraries_agree_at_common_temps(self):
        """
        Test that our equations agree with both MetPy and PsychroLib.
        
        Uses temperatures where all libraries have reference values.
        """
        temps_c = np.array([0, 10, 20, 25, 30])
        temps_k = temps_c + 273.15
        
        # Initialize our equations
        goff = GoffGratchEquation(surface_type=SurfaceType.WATER)
        hyland = HylandWexlerEquation(surface_type=SurfaceType.WATER)
        bolton = BoltonEquation()
        
        for temp_c, temp_k in zip(temps_c, temps_k):
            # Get MetPy result
            metpy_result = mpcalc.saturation_vapor_pressure(temp_c * units.degC)
            metpy_hpa = metpy_result.to('hPa').magnitude
            
            # Get PsychroLib result
            psychro_pa = psy.GetSatVapPres(temp_c)
            psychro_hpa = psychro_pa / 100.0
            
            # Get our results
            goff_result = goff.calculate(temp_k)
            hyland_result = hyland.calculate(temp_k)
            bolton_result = bolton.calculate(temp_k)
            
            # All should agree with MetPy
            assert_allclose(goff_result, metpy_hpa, rtol=0.015, 
                           err_msg=f"Goff-Gratch vs MetPy at {temp_c}°C")
            assert_allclose(hyland_result, metpy_hpa, rtol=0.015,
                           err_msg=f"Hyland-Wexler vs MetPy at {temp_c}°C")
            assert_allclose(bolton_result, metpy_hpa, rtol=0.015,
                           err_msg=f"Bolton vs MetPy at {temp_c}°C")
            
            # Hyland-Wexler should match PsychroLib very closely
            assert_allclose(hyland_result, psychro_hpa, rtol=0.005,
                           err_msg=f"Hyland-Wexler vs PsychroLib at {temp_c}°C")
    
    @pytest.mark.skipif(not METPY_AVAILABLE, reason="MetPy required")
    def test_vectorized_vs_metpy(self):
        """Test that vectorized calculations match MetPy."""
        temps_c = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
        temps_k = temps_c + 273.15
        
        # MetPy vectorized results
        metpy_results = mpcalc.saturation_vapor_pressure(temps_c * units.degC)
        metpy_hpa = metpy_results.to('hPa').magnitude
        
        # Our vectorized results
        goff = GoffGratchEquation(surface_type=SurfaceType.WATER)
        our_results = goff.calculate(temps_k)
        
        assert_allclose(our_results, metpy_hpa, rtol=0.01)
    
    @pytest.mark.skipif(not (METPY_AVAILABLE and PSYCHROLIB_AVAILABLE),
                       reason="MetPy and PsychroLib required")
    def test_ice_phase_agreement(self):
        """Test that ice calculations agree across libraries."""
        temps_c = np.array([-30, -20, -10, -5])
        temps_k = temps_c + 273.15
        
        goff_ice = GoffGratchEquation(surface_type=SurfaceType.ICE)
        hyland_ice = HylandWexlerEquation(surface_type=SurfaceType.ICE)
        
        for temp_c, temp_k in zip(temps_c, temps_k):
            # MetPy
            metpy_result = mpcalc.saturation_vapor_pressure(temp_c * units.degC)
            metpy_hpa = metpy_result.to('hPa').magnitude
            
            # PsychroLib (automatically uses ice for T < 0)
            psychro_pa = psy.GetSatVapPres(temp_c)
            psychro_hpa = psychro_pa / 100.0
            
            # Our results
            goff_result = goff_ice.calculate(temp_k)
            hyland_result = hyland_ice.calculate(temp_k)
            
            # Check agreement
            assert_allclose(goff_result, metpy_hpa, rtol=0.02,
                           err_msg=f"Goff-Gratch ice vs MetPy at {temp_c}°C")
            assert_allclose(hyland_result, psychro_hpa, rtol=0.01,
                           err_msg=f"Hyland-Wexler ice vs PsychroLib at {temp_c}°C")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_dimensional_array_input(self):
        """Test that 0-d arrays are handled correctly."""
        bolton = BoltonEquation()
        temp_k = np.array(293.15)  # 0-d array
        
        result = bolton.calculate(temp_k)
        assert isinstance(result, (float, np.floating))
    
    def test_multi_dimensional_array_shape(self):
        """Test that multi-dimensional arrays preserve shape."""
        bolton = BoltonEquation()
        temps_3d = np.ones((2, 3, 4)) * 293.15
        
        result = bolton.calculate(temps_3d)
        assert result.shape == temps_3d.shape
    
    def test_single_element_array(self):
        """Test single-element array behaves correctly."""
        bolton = BoltonEquation()
        temps = np.array([293.15])
        
        result = bolton.calculate(temps)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
    
    def test_empty_regions_in_automatic_mode(self):
        """Test automatic mode when some temperature regions are empty."""
        goff = GoffGratchEquation(surface_type=SurfaceType.AUTOMATIC)
        
        # Only temperatures above freezing
        temps = np.array([280.0, 285.0, 290.0])
        result = goff.calculate(temps)
        
        assert result.shape == temps.shape
        assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])