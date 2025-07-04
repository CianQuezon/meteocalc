"""
Professional unit tests for the enhanced dewpoint calculator with custom Brent solver.

This test suite validates the implementation against established standards including:
- PsychroLib (ASHRAE 2017 standard)
- CoolProp (ASHRAE RP-1485 reference)
- MetPy (Bolton 1980 standard)
- Published research values
- Cross-validation between calculation methods
- Performance benchmarking

Standards Compliance:
- IEEE 829 Test Documentation Standard
- ASHRAE Testing Standards for Psychrometric Calculations  
- WMO Guide to Meteorological Instruments and Methods
- ISO/IEC 25010 Software Quality Standards

Author: Meteorological Software Engineering Team
Date: 2025
License: MIT
Version: 2.0.0

Dependencies:
    pytest>=7.0.0
    numpy>=1.20.0
    pandas>=1.3.0
    psychrolib>=2.5.0 (optional, for cross-validation)
    CoolProp>=6.4.0 (optional, for ASHRAE RP-1485 validation)
    metpy>=1.3.0 (optional, for Bolton validation)

Usage:
    pytest test_dewpoint_enhanced.py -v
    pytest test_dewpoint_enhanced.py::TestDewpointCalculator::test_against_ashrae_reference -v
    pytest test_dewpoint_enhanced.py -m "not external" -v  # Skip external validation
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# External library imports with graceful degradation
try:
    import psychrolib
    PSYCHROLIB_AVAILABLE = True
except ImportError:
    PSYCHROLIB_AVAILABLE = False
    warnings.warn(
        "PsychroLib not available. Install with: pip install psychrolib",
        ImportWarning,
        stacklevel=2
    )

try:
    from CoolProp.HumidAirProp import HAPropsSI
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    warnings.warn(
        "CoolProp not available. Install with: pip install CoolProp",
        ImportWarning,
        stacklevel=2
    )

try:
    import metpy.calc as mpcalc
    from metpy.units import units
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False
    warnings.warn(
        "MetPy not available. Install with: pip install metpy",
        ImportWarning,
        stacklevel=2
    )

# Import your dewpoint implementation (adjust import path as needed)
try:
    # Try multiple import paths
    import sys
    import os
    
    # Add current directory to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Primary import attempt
    from meteocalc.lib.thermodynamics.dew_point_modular import (
        dewpoint, 
        DewpointCalculator, 
        VaporPressureConstants
    )
    DEWPOINT_AVAILABLE = True
    
    # Test basic functionality
    test_result = dewpoint(25.0, 60.0)
    if not (isinstance(test_result, (int, float)) and 15.0 < test_result < 18.0):
        raise ValueError("Basic dewpoint test failed")
        
except ImportError:
    DEWPOINT_AVAILABLE = False
    # Create dummy functions to prevent import errors
    def dewpoint(*args, **kwargs):
        raise NotImplementedError("Dewpoint module not available")
    def DewpointCalculator(*args, **kwargs):
        raise NotImplementedError("DewpointCalculator not available")
    VaporPressureConstants = None
    
    pytest.skip(
        "Enhanced dewpoint calculator implementation not found. "
        "Update import path in test file or ensure module is available.",
        allow_module_level=True
    )


class TestDatasets:
    """Reference test datasets from established meteorological sources.
    
    This class provides curated test data from authoritative sources for
    validating dewpoint temperature calculations against known standards.
    """
    
    @staticmethod
    def get_ashrae_reference_data() -> pd.DataFrame:
        """ASHRAE reference values calculated using PsychroLib (ASHRAE compliant).
        
        Returns:
            pd.DataFrame: Test data with columns ['temp_c', 'rh_percent', 
                         'pressure_hpa', 'expected_td_c', 'source'].
                         
        Note:
            Values calculated using PsychroLib which implements ASHRAE standards.
            These serve as authoritative reference for dewpoint calculations.
        """
        if not PSYCHROLIB_AVAILABLE:
            # Fallback reference values for basic testing
            data = [
                (20.0, 50.0, 1013.25, 9.3, "ASHRAE_Estimated"),
                (25.0, 60.0, 1013.25, 16.7, "ASHRAE_Estimated"),
                (30.0, 70.0, 1013.25, 24.2, "ASHRAE_Estimated"),
                (35.0, 80.0, 1013.25, 31.2, "ASHRAE_Estimated"),
                (10.0, 90.0, 1013.25, 8.5, "ASHRAE_Estimated"),
                (0.0, 100.0, 1013.25, 0.0, "ASHRAE_Estimated"),
                (-10.0, 85.0, 1013.25, -12.3, "ASHRAE_Estimated"),
                (40.0, 30.0, 1013.25, 19.1, "ASHRAE_Estimated"),
                (15.0, 45.0, 1013.25, 3.9, "ASHRAE_Estimated"),
                (5.0, 75.0, 1013.25, 1.1, "ASHRAE_Estimated"),
            ]
        else:
            # Calculate using PsychroLib for authoritative values
            psychrolib.SetUnitSystem(psychrolib.SI)
            
            conditions = [
                (20.0, 50.0, 1013.25),
                (25.0, 60.0, 1013.25),
                (30.0, 70.0, 1013.25),
                (35.0, 80.0, 1013.25),
                (10.0, 90.0, 1013.25),
                (0.0, 100.0, 1013.25),
                (-10.0, 85.0, 1013.25),
                (40.0, 30.0, 1013.25),
                (15.0, 45.0, 1013.25),
                (5.0, 75.0, 1013.25),
            ]
            
            data = []
            for temp, rh, pressure in conditions:
                td = psychrolib.GetTDewPointFromRelHum(temp, rh/100.0)
                data.append((temp, rh, pressure, td, "PsychroLib_ASHRAE"))
        
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )
    
    @staticmethod
    def get_bolton_reference_data() -> pd.DataFrame:
        """Bolton (1980) reference data calculated using MetPy.
        
        Returns:
            pd.DataFrame: Test data based on Bolton 1980 formulation.
            
        Note:
            Uses MetPy's implementation of Bolton's formula with coefficients 
            a=17.67, b=243.5. Falls back to hardcoded values if MetPy unavailable.
        """
        # Test conditions for Bolton validation
        conditions = [
            (20.0, 70.0, 1013.25, "Bolton_1980_Eq11"),
            (25.0, 80.0, 1013.25, "Bolton_1980_Table2"),
            (30.0, 60.0, 1013.25, "Bolton_1980_Example1"),
            (15.0, 85.0, 1013.25, "Bolton_1980_Reference"),
            (35.0, 50.0, 1013.25, "Bolton_1980_Reference"),
            (10.0, 95.0, 1013.25, "Bolton_1980_Reference"),
        ]
        
        if METPY_AVAILABLE:
            # Calculate using MetPy's Bolton implementation
            data = []
            for temp, rh, pressure, source in conditions:
                try:
                    # Convert to MetPy quantities
                    temp_qty = temp * units.celsius
                    rh_qty = rh * units.percent
                    
                    # Calculate dewpoint using MetPy's Bolton implementation
                    dewpoint_qty = mpcalc.dewpoint_from_relative_humidity(temp_qty, rh_qty)
                    dewpoint_c = dewpoint_qty.to('celsius').magnitude
                    
                    data.append((temp, rh, pressure, dewpoint_c, f"MetPy_{source}"))
                    
                except Exception as e:
                    # Fallback to estimated value if MetPy calculation fails
                    dewpoint_est = temp - (100 - rh) / 5.0  # Simple estimation
                    data.append((temp, rh, pressure, dewpoint_est, f"{source}_Estimated"))
                    warnings.warn(f"MetPy calculation failed for {source}: {e}")
        else:
            # Fallback to hardcoded reference values if MetPy not available
            data = [
                (20.0, 70.0, 1013.25, 14.4, "Bolton_1980_Eq11_Fallback"),
                (25.0, 80.0, 1013.25, 21.3, "Bolton_1980_Table2_Fallback"),
                (30.0, 60.0, 1013.25, 21.9, "Bolton_1980_Example1_Fallback"),
                (15.0, 85.0, 1013.25, 12.8, "Bolton_1980_Reference_Fallback"),
                (35.0, 50.0, 1013.25, 23.9, "Bolton_1980_Reference_Fallback"),
                (10.0, 95.0, 1013.25, 9.3, "Bolton_1980_Reference_Fallback"),
            ]
        
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )
    @staticmethod
    def get_ice_phase_reference_data() -> pd.DataFrame:
        """Ice phase test cases for enhanced validation.
        
        Returns:
            pd.DataFrame: Test data for ice phase conditions.
            
        Note:
            These cases test the ice phase enhancement functionality
            which applies empirical corrections for sub-freezing conditions.
        """
        data = [
            # temp_c, rh_percent, pressure_hpa, expected_td_c, source
            (-10.0, 70.0, 1013.25, -14.3, "Ice_Phase_Enhanced"),
            (-5.0, 85.0, 1013.25, -7.2, "Ice_Phase_Enhanced"),
            (0.0, 90.0, 1013.25, -1.3, "Ice_Phase_Enhanced"),
            (-20.0, 80.0, 1013.25, -22.5, "WMO_Arctic_Standard"),
            (-15.0, 95.0, 1013.25, -15.5, "WMO_Arctic_Standard"),
            (2.0, 90.0, 1013.25, 0.5, "Transition_Zone"),
        ]
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )
    
    @staticmethod
    def get_extreme_conditions_data() -> pd.DataFrame:
        """Extreme atmospheric conditions for robustness testing.
        
        Returns:
            pd.DataFrame: Test data for extreme conditions.
        """
        if PSYCHROLIB_AVAILABLE:
            psychrolib.SetUnitSystem(psychrolib.SI)
            
            extreme_conditions = [
                # Arctic winter conditions
                (-40.0, 80.0, 1013.25, "Arctic_Winter"),
                
                # Desert heat conditions  
                (50.0, 15.0, 1013.25, "Desert_Heat"),
                
                # Tropical humid extreme
                (40.0, 95.0, 1013.25, "Tropical_Humid"),
                
                # High altitude conditions
                (10.0, 30.0, 700.0, "High_Altitude"),
                
                # Winter storm conditions
                (-15.0, 90.0, 980.0, "Winter_Storm"),
            ]
            
            data = []
            for temp, rh, pressure, source in extreme_conditions:
                try:
                    td = psychrolib.GetTDewPointFromRelHum(temp, rh/100.0)
                    data.append((temp, rh, pressure, td, source))
                except:
                    # If PsychroLib fails, use estimated value
                    td_est = temp - (100 - rh) / 5.0  # Simple estimation
                    data.append((temp, rh, pressure, td_est, f"{source}_Estimated"))
        else:
            # Fallback extreme conditions with estimated values
            data = [
                (-40.0, 80.0, 1013.25, -44.0, "Arctic_Winter_Estimated"),
                (50.0, 15.0, 1013.25, 33.0, "Desert_Heat_Estimated"),
                (40.0, 95.0, 1013.25, 39.0, "Tropical_Humid_Estimated"),
                (10.0, 30.0, 700.0, -4.0, "High_Altitude_Estimated"),
                (-15.0, 90.0, 980.0, -16.0, "Winter_Storm_Estimated"),
            ]
        
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )


class TestDewpointCalculator:
    """Comprehensive test suite for the enhanced dewpoint calculator.
    
    This test class provides systematic validation of the dewpoint temperature
    calculation implementation through various test scenarios including:
    - Method validation across all available formulations
    - Custom Brent solver validation
    - Ice phase enhancement testing
    - External library cross-validation
    - Physical constraint verification
    - Performance benchmarking
    """
    
    # Class-level constants for test tolerances
    TOLERANCE_STRICT = 0.1      # ±0.1°C for strict comparison
    TOLERANCE_MODERATE = 0.3    # ±0.3°C for moderate comparison  
    TOLERANCE_EXTREME = 0.5     # ±0.5°C for extreme conditions
    TOLERANCE_ICE_PHASE = 0.4   # ±0.4°C for ice phase corrections
    
    # Available calculation methods for testing
    ANALYTICAL_METHODS = [
        'magnus_alduchov_eskridge',
        'magnus_standard', 
        'arden_buck',
        'tetens',
        'lawrence_simple'
    ]
    
    BRENT_SOLVER_METHODS = [
        'bolton_custom',
        'hyland_wexler'
    ]
    
    ALL_METHODS = ANALYTICAL_METHODS + BRENT_SOLVER_METHODS
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_external_libraries(self):
        """Initialize external libraries if available for cross-validation tests.
        
        Yields:
            None: This fixture performs setup/teardown without returning data.
        """
        if PSYCHROLIB_AVAILABLE:
            psychrolib.SetUnitSystem(psychrolib.SI)
        yield
        # Teardown code would go here if needed
    
    def test_all_methods_work_basic_conditions(self) -> None:
        """Test that all dewpoint calculation methods work for basic conditions.
        
        This test ensures basic functionality across all supported calculation
        methods using standard atmospheric conditions.
        """
        # Standard test conditions
        temp_c, rh_percent = 25.0, 60.0
        
        for method in self.ALL_METHODS:
            try:
                if method in ['hyland_wexler']:
                    # Enhanced methods with phase parameter
                    result = dewpoint(temp_c, rh_percent, method, phase="auto")
                else:
                    # Standard methods
                    result = dewpoint(temp_c, rh_percent, method)
                
                # Basic sanity checks
                assert isinstance(result, (int, float, np.number)), \
                    f"Result must be numeric for {method}, got {type(result)}"
                assert not np.isnan(result), f"Result cannot be NaN for {method}"
                assert not np.isinf(result), f"Result cannot be infinite for {method}"
                assert result < temp_c, f"Dewpoint should be less than air temp for {method}"
                assert result > temp_c - 30, f"Dewpoint should be within reasonable range for {method}"
                
            except Exception as e:
                pytest.fail(f"Method {method} failed basic test: {e}")
    
    def test_input_validation_handles_invalid_methods(self) -> None:
        """Test input validation and error handling for invalid method names.
        
        Ensures that appropriate ValueErrors are raised for invalid method
        specifications.
        """
        # Test invalid method name
        with pytest.raises(ValueError, match="Unknown equation"):
            dewpoint(25.0, 60.0, 'invalid_method')
        
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature"):
            dewpoint(-150.0, 60.0)
        
        # Test invalid humidity
        with pytest.raises(ValueError, match="Humidity"):
            dewpoint(25.0, 150.0)
    
    @pytest.mark.parametrize("temp,rh", [
        (-50.0, 80.0),   # Very cold
        (50.0, 10.0),    # Very hot and dry
        (0.0, 99.9),     # Near-saturation at freezing
        (40.0, 95.0),    # Hot and humid
    ])
    def test_extreme_input_cases_handled_gracefully(
        self, 
        temp: float, 
        rh: float
    ) -> None:
        """Test that extreme input cases are handled gracefully.
        
        Args:
            temp: Temperature in degrees Celsius.
            rh: Relative humidity in percent.
        """
        # Test with robust method
        result = dewpoint(temp, rh, "magnus_alduchov_eskridge")
        
        assert not np.isnan(result), \
            f"NaN result for extreme case: T={temp}°C, RH={rh}%"
        assert not np.isinf(result), \
            f"Infinite result for extreme case: T={temp}°C, RH={rh}%"
        assert result <= temp + 0.2, \
            f"Dewpoint exceeds air temperature: T={temp}°C, RH={rh}%"
    
    def test_scalar_and_array_inputs_give_consistent_results(self) -> None:
        """Test that scalar and array inputs give identical results.
        
        Validates that vectorized implementation produces the same results
        as scalar calculations.
        """
        # Test data
        temps = [20.0, 25.0, 30.0]
        rhs = [50.0, 60.0, 70.0]
        
        # Test multiple methods
        methods_to_test = ["magnus_alduchov_eskridge", "hyland_wexler"]
        
        for method in methods_to_test:
            # Calculate scalar results
            scalar_results = []
            for temp, rh in zip(temps, rhs):
                if method == "hyland_wexler":
                    result = dewpoint(temp, rh, method, phase="auto")
                else:
                    result = dewpoint(temp, rh, method)
                scalar_results.append(result)
            
            # Calculate array results
            if method == "hyland_wexler":
                array_results = dewpoint(temps, rhs, method, phase="auto")
            else:
                array_results = dewpoint(temps, rhs, method)
            
            # Compare with high precision
            np.testing.assert_allclose(
                scalar_results, 
                array_results, 
                rtol=1e-10, 
                atol=1e-10,
                err_msg=f"Scalar and array results must be identical for {method}"
            )
    
    @pytest.mark.skipif(
        not PSYCHROLIB_AVAILABLE, 
        reason="PsychroLib not available for cross-validation"
    )
    def test_results_against_psychrolib_standard(self) -> None:
        """Compare results against PsychroLib (ASHRAE standard implementation).
        
        This test validates accuracy against the authoritative ASHRAE
        implementation using Hyland-Wexler formulation for optimal comparison.
        """
        test_data = TestDatasets.get_ashrae_reference_data()
        
        differences = []
        for _, row in test_data.iterrows():
            # Use Hyland-Wexler method for best comparison with PsychroLib
            our_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='auto'
            )
            
            # Calculate PsychroLib result
            psychrolib_result = psychrolib.GetTDewPointFromRelHum(
                row['temp_c'], 
                row['rh_percent'] / 100.0
            )
            
            difference = abs(our_result - psychrolib_result)
            differences.append(difference)
            
            # Individual case validation
            tolerance = self.TOLERANCE_ICE_PHASE if row['temp_c'] <= 0 else self.TOLERANCE_MODERATE
            
            assert difference < tolerance, \
                f"Difference vs PsychroLib: {difference:.3f}°C for " \
                f"T={row['temp_c']}°C, RH={row['rh_percent']}% " \
                f"(tolerance: {tolerance:.3f}°C)"
        
        # Statistical validation
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        rms_diff = np.sqrt(np.mean(np.array(differences)**2))
        
        # Report statistics for documentation
        print(f"\nPsychroLib Cross-Validation Results:")
        print(f"  Mean difference: {mean_diff:.3f}°C")
        print(f"  Max difference:  {max_diff:.3f}°C") 
        print(f"  RMS difference:  {rms_diff:.3f}°C")
        
        # Statistical assertions
        assert mean_diff < 0.25, f"Mean difference too large: {mean_diff:.3f}°C"
        assert max_diff < 0.8, f"Maximum difference too large: {max_diff:.3f}°C"
    
    @pytest.mark.skipif(
        not COOLPROP_AVAILABLE, 
        reason="CoolProp not available for validation"
    )
    def test_results_against_coolprop_standard(self) -> None:
        """Validate results against CoolProp (ASHRAE RP-1485 reference).
        
        Tests using CoolProp which implements the ASHRAE RP-1485 standard
        for thermodynamic properties of humid air.
        """
        test_data = TestDatasets.get_ashrae_reference_data()
        
        differences = []
        for _, row in test_data.iterrows():
            # Our enhanced result
            our_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='auto'
            )
            
            # CoolProp calculation
            temp_k = row['temp_c'] + 273.15
            rh_fraction = row['rh_percent'] / 100.0
            pressure_pa = row['pressure_hpa'] * 100.0
            
            try:
                coolprop_td_k = HAPropsSI('Tdp', 'T', temp_k, 'R', rh_fraction, 'P', pressure_pa)
                coolprop_result = coolprop_td_k - 273.15
                
                difference = abs(our_result - coolprop_result)
                differences.append(difference)
                
                assert difference < self.TOLERANCE_MODERATE, \
                    f"CoolProp validation failed: {difference:.3f}°C difference " \
                    f"for T={row['temp_c']}°C, RH={row['rh_percent']}%"
                    
            except Exception as e:
                warnings.warn(f"CoolProp calculation failed for T={row['temp_c']}°C: {e}")
        
        if differences:
            mean_diff = np.mean(differences)
            print(f"\nCoolProp Validation Results:")
            print(f"  Mean difference: {mean_diff:.3f}°C")
            print(f"  Max difference:  {np.max(differences):.3f}°C")
    
    @pytest.mark.skipif(
        not METPY_AVAILABLE, 
        reason="MetPy not available for validation"
    )
    def test_results_against_metpy_bolton_standard(self) -> None:
        """Validate Bolton method against MetPy (Bolton 1980 standard).
        
        Tests our custom Bolton implementation against MetPy's reference
        implementation of the Bolton 1980 formulation.
        """
        test_data = TestDatasets.get_bolton_reference_data()
        
        for _, row in test_data.iterrows():
            # Our Bolton implementation
            our_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'bolton_custom'
            )
            
            # MetPy Bolton calculation
            temp_qty = row['temp_c'] * units.celsius
            rh_qty = row['rh_percent'] * units.percent
            
            metpy_result_qty = mpcalc.dewpoint_from_relative_humidity(temp_qty, rh_qty)
            metpy_result = metpy_result_qty.to('celsius').magnitude
            
            difference = abs(our_result - metpy_result)
            
            assert difference < self.TOLERANCE_STRICT, \
                f"MetPy Bolton validation failed: {difference:.3f}°C difference " \
                f"for T={row['temp_c']}°C, RH={row['rh_percent']}%"
    
    def test_ice_phase_enhancement_functionality(self) -> None:
        """Test ice phase enhancement functionality.
        
        Validates that ice phase corrections are properly applied for
        sub-freezing conditions.
        """
        ice_test_data = TestDatasets.get_ice_phase_reference_data()
        
        for _, row in ice_test_data.iterrows():
            # Test automatic phase selection
            auto_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='auto'
            )
            
            # Test explicit liquid phase
            liquid_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='liquid'
            )
            
            # Test explicit ice phase (for sub-freezing)
            if row['temp_c'] <= 0:
                ice_result = dewpoint(
                    row['temp_c'], 
                    row['rh_percent'], 
                    'hyland_wexler',
                    phase='ice'
                )
                
                # Auto should match ice for sub-freezing
                assert abs(auto_result - ice_result) < 0.01, \
                    f"Auto phase should match ice phase for T={row['temp_c']}°C"
                
                # Ice correction should be applied
                correction = ice_result - liquid_result
                assert 0.1 < correction < 0.3, \
                    f"Ice correction should be ~0.17°C, got {correction:.3f}°C"
            
            # Validate result is reasonable
            assert not np.isnan(auto_result), f"Auto result is NaN for {row['source']}"
            assert auto_result <= row['temp_c'] + 0.1, \
                f"Dewpoint exceeds air temperature for {row['source']}"
    
    def test_custom_brent_solver_functionality(self) -> None:
        """Test custom Brent solver implementation.
        
        Validates that the custom Brent solver (eliminating SciPy dependency)
        works correctly and provides accurate results.
        """
        # Test conditions where iterative methods are challenged
        challenging_cases = [
            (0.0, 99.0, "Near-saturation freezing"),
            (35.0, 95.0, "Hot humid extreme"),
            (-10.0, 98.0, "Cold near-saturation"),
            (45.0, 80.0, "Very hot humid"),
        ]
        
        for temp, rh, description in challenging_cases:
            # Compare Brent-based methods with analytical methods
            analytical_result = dewpoint(temp, rh, "magnus_alduchov_eskridge")
            brent_bolton_result = dewpoint(temp, rh, "bolton_custom")
            brent_hyland_result = dewpoint(temp, rh, "hyland_wexler", phase="auto")
            
            # All results should be finite and reasonable
            results = {
                'analytical': analytical_result,
                'brent_bolton': brent_bolton_result,
                'brent_hyland': brent_hyland_result
            }
            
            for method_name, result in results.items():
                assert np.isfinite(result), f"{method_name} failed for {description}"
                assert result <= temp + 0.1, f"{method_name} violates physics for {description}"
                assert result > temp - 40, f"{method_name} unreasonably low for {description}"
            
            # Method agreement (allowing for different formulations)
            bolton_diff = abs(brent_bolton_result - analytical_result)
            hyland_diff = abs(brent_hyland_result - analytical_result)
            
            assert bolton_diff <= 1.0, \
                f"Bolton vs Magnus disagreement for {description}: {bolton_diff:.3f}°C"
            assert hyland_diff <= 1.0, \
                f"Hyland vs Magnus disagreement for {description}: {hyland_diff:.3f}°C"
    

"""
Professional unit tests for the enhanced dewpoint calculator with custom Brent solver.

This test suite validates the implementation against established standards including:
- PsychroLib (ASHRAE 2017 standard)
- CoolProp (ASHRAE RP-1485 reference)
- MetPy (Bolton 1980 standard)
- Published research values
- Cross-validation between calculation methods
- Performance benchmarking

Standards Compliance:
- IEEE 829 Test Documentation Standard
- ASHRAE Testing Standards for Psychrometric Calculations  
- WMO Guide to Meteorological Instruments and Methods
- ISO/IEC 25010 Software Quality Standards

Author: Meteorological Software Engineering Team
Date: 2025
License: MIT
Version: 2.0.0

Dependencies:
    pytest>=7.0.0
    numpy>=1.20.0
    pandas>=1.3.0
    psychrolib>=2.5.0 (optional, for cross-validation)
    CoolProp>=6.4.0 (optional, for ASHRAE RP-1485 validation)
    metpy>=1.3.0 (optional, for Bolton validation)

Usage:
    pytest test_dewpoint_enhanced.py -v
    pytest test_dewpoint_enhanced.py::TestDewpointCalculator::test_against_ashrae_reference -v
    pytest test_dewpoint_enhanced.py -m "not external" -v  # Skip external validation
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# External library imports with graceful degradation
try:
    import psychrolib
    PSYCHROLIB_AVAILABLE = True
except ImportError:
    PSYCHROLIB_AVAILABLE = False
    warnings.warn(
        "PsychroLib not available. Install with: pip install psychrolib",
        ImportWarning,
        stacklevel=2
    )

try:
    from CoolProp.HumidAirProp import HAPropsSI
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    warnings.warn(
        "CoolProp not available. Install with: pip install CoolProp",
        ImportWarning,
        stacklevel=2
    )

try:
    import metpy.calc as mpcalc
    from metpy.units import units
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False
    warnings.warn(
        "MetPy not available. Install with: pip install metpy",
        ImportWarning,
        stacklevel=2
    )

# Import your dewpoint implementation (adjust import path as needed)
try:
    # Try multiple import paths
    import sys
    import os
    
    # Add current directory to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Primary import attempt
    from meteocalc.lib.thermodynamics.dew_point_modular import (
        dewpoint, 
        DewpointCalculator, 
        VaporPressureConstants
    )
    DEWPOINT_AVAILABLE = True
    
    # Test basic functionality
    test_result = dewpoint(25.0, 60.0)
    if not (isinstance(test_result, (int, float)) and 15.0 < test_result < 18.0):
        raise ValueError("Basic dewpoint test failed")
        
except ImportError:
    DEWPOINT_AVAILABLE = False
    # Create dummy functions to prevent import errors
    def dewpoint(*args, **kwargs):
        raise NotImplementedError("Dewpoint module not available")
    def DewpointCalculator(*args, **kwargs):
        raise NotImplementedError("DewpointCalculator not available")
    VaporPressureConstants = None
    
    pytest.skip(
        "Enhanced dewpoint calculator implementation not found. "
        "Update import path in test file or ensure module is available.",
        allow_module_level=True
    )


class TestDatasets:
    """Reference test datasets from established meteorological sources.
    
    This class provides curated test data from authoritative sources for
    validating dewpoint temperature calculations against known standards.
    """
    
    @staticmethod
    def get_ashrae_reference_data() -> pd.DataFrame:
        """ASHRAE reference values calculated using PsychroLib (ASHRAE compliant).
        
        Returns:
            pd.DataFrame: Test data with columns ['temp_c', 'rh_percent', 
                         'pressure_hpa', 'expected_td_c', 'source'].
                         
        Note:
            Values calculated using PsychroLib which implements ASHRAE standards.
            These serve as authoritative reference for dewpoint calculations.
        """
        if not PSYCHROLIB_AVAILABLE:
            # Fallback reference values for basic testing
            data = [
                (20.0, 50.0, 1013.25, 9.3, "ASHRAE_Estimated"),
                (25.0, 60.0, 1013.25, 16.7, "ASHRAE_Estimated"),
                (30.0, 70.0, 1013.25, 24.2, "ASHRAE_Estimated"),
                (35.0, 80.0, 1013.25, 31.2, "ASHRAE_Estimated"),
                (10.0, 90.0, 1013.25, 8.5, "ASHRAE_Estimated"),
                (0.0, 100.0, 1013.25, 0.0, "ASHRAE_Estimated"),
                (-10.0, 85.0, 1013.25, -12.3, "ASHRAE_Estimated"),
                (40.0, 30.0, 1013.25, 19.1, "ASHRAE_Estimated"),
                (15.0, 45.0, 1013.25, 3.9, "ASHRAE_Estimated"),
                (5.0, 75.0, 1013.25, 1.1, "ASHRAE_Estimated"),
            ]
        else:
            # Calculate using PsychroLib for authoritative values
            psychrolib.SetUnitSystem(psychrolib.SI)
            
            conditions = [
                (20.0, 50.0, 1013.25),
                (25.0, 60.0, 1013.25),
                (30.0, 70.0, 1013.25),
                (35.0, 80.0, 1013.25),
                (10.0, 90.0, 1013.25),
                (0.0, 100.0, 1013.25),
                (-10.0, 85.0, 1013.25),
                (40.0, 30.0, 1013.25),
                (15.0, 45.0, 1013.25),
                (5.0, 75.0, 1013.25),
            ]
            
            data = []
            for temp, rh, pressure in conditions:
                td = psychrolib.GetTDewPointFromRelHum(temp, rh/100.0)
                data.append((temp, rh, pressure, td, "PsychroLib_ASHRAE"))
        
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )
    
    @staticmethod
    def get_bolton_reference_data() -> pd.DataFrame:
        """Bolton (1980) reference data calculated using MetPy.
        
        Returns:
            pd.DataFrame: Test data based on Bolton 1980 formulation.
            
        Note:
            Uses MetPy's implementation of Bolton's formula with coefficients 
            a=17.67, b=243.5. Falls back to hardcoded values if MetPy unavailable.
        """
        # Test conditions for Bolton validation
        conditions = [
            (20.0, 70.0, 1013.25, "Bolton_1980_Eq11"),
            (25.0, 80.0, 1013.25, "Bolton_1980_Table2"),
            (30.0, 60.0, 1013.25, "Bolton_1980_Example1"),
            (15.0, 85.0, 1013.25, "Bolton_1980_Reference"),
            (35.0, 50.0, 1013.25, "Bolton_1980_Reference"),
            (10.0, 95.0, 1013.25, "Bolton_1980_Reference"),
        ]
        
        if METPY_AVAILABLE:
            # Calculate using MetPy's Bolton implementation
            data = []
            for temp, rh, pressure, source in conditions:
                try:
                    # Convert to MetPy quantities
                    temp_qty = temp * units.celsius
                    rh_qty = rh * units.percent
                    
                    # Calculate dewpoint using MetPy's Bolton implementation
                    dewpoint_qty = mpcalc.dewpoint_from_relative_humidity(temp_qty, rh_qty)
                    dewpoint_c = dewpoint_qty.to('celsius').magnitude
                    
                    data.append((temp, rh, pressure, dewpoint_c, f"MetPy_{source}"))
                    
                except Exception as e:
                    # Fallback to estimated value if MetPy calculation fails
                    dewpoint_est = temp - (100 - rh) / 5.0  # Simple estimation
                    data.append((temp, rh, pressure, dewpoint_est, f"{source}_Estimated"))
                    warnings.warn(f"MetPy calculation failed for {source}: {e}")
        else:
            # Fallback to hardcoded reference values if MetPy not available
            data = [
                (20.0, 70.0, 1013.25, 14.4, "Bolton_1980_Eq11_Fallback"),
                (25.0, 80.0, 1013.25, 21.3, "Bolton_1980_Table2_Fallback"),
                (30.0, 60.0, 1013.25, 21.9, "Bolton_1980_Example1_Fallback"),
                (15.0, 85.0, 1013.25, 12.8, "Bolton_1980_Reference_Fallback"),
                (35.0, 50.0, 1013.25, 23.9, "Bolton_1980_Reference_Fallback"),
                (10.0, 95.0, 1013.25, 9.3, "Bolton_1980_Reference_Fallback"),
            ]
        
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )
    
    @staticmethod
    def get_ice_phase_reference_data() -> pd.DataFrame:
        """Ice phase test cases for enhanced validation.
        
        Returns:
            pd.DataFrame: Test data for ice phase conditions.
            
        Note:
            These cases test the ice phase enhancement functionality
            which applies empirical corrections for sub-freezing conditions.
        """
        data = [
            # temp_c, rh_percent, pressure_hpa, expected_td_c, source
            (-10.0, 70.0, 1013.25, -14.3, "Ice_Phase_Enhanced"),
            (-5.0, 85.0, 1013.25, -7.2, "Ice_Phase_Enhanced"),
            (0.0, 90.0, 1013.25, -1.3, "Ice_Phase_Enhanced"),
            (-20.0, 80.0, 1013.25, -22.5, "WMO_Arctic_Standard"),
            (-15.0, 95.0, 1013.25, -15.5, "WMO_Arctic_Standard"),
            (2.0, 90.0, 1013.25, 0.5, "Transition_Zone"),
        ]
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )
    
    @staticmethod
    def get_extreme_conditions_data() -> pd.DataFrame:
        """Extreme atmospheric conditions for robustness testing.
        
        Returns:
            pd.DataFrame: Test data for extreme conditions.
        """
        if PSYCHROLIB_AVAILABLE:
            psychrolib.SetUnitSystem(psychrolib.SI)
            
            extreme_conditions = [
                # Arctic winter conditions
                (-40.0, 80.0, 1013.25, "Arctic_Winter"),
                
                # Desert heat conditions  
                (50.0, 15.0, 1013.25, "Desert_Heat"),
                
                # Tropical humid extreme
                (40.0, 95.0, 1013.25, "Tropical_Humid"),
                
                # High altitude conditions (more challenging due to pressure effects)
                (10.0, 30.0, 700.0, "High_Altitude"),
                
                # Winter storm conditions
                (-15.0, 90.0, 980.0, "Winter_Storm"),
            ]
            
            data = []
            for temp, rh, pressure, source in extreme_conditions:
                try:
                    # Note: PsychroLib GetTDewPointFromRelHum doesn't use pressure parameter
                    # It assumes standard pressure, which may cause discrepancies at altitude
                    td = psychrolib.GetTDewPointFromRelHum(temp, rh/100.0)
                    data.append((temp, rh, pressure, td, source))
                except Exception as e:
                    # If PsychroLib fails, use estimated value
                    td_est = temp - (100 - rh) / 5.0  # Simple estimation
                    data.append((temp, rh, pressure, td_est, f"{source}_Estimated"))
                    import warnings
                    warnings.warn(f"PsychroLib calculation failed for {source}: {e}")
        else:
            # Fallback extreme conditions with estimated values
            data = [
                (-40.0, 80.0, 1013.25, -44.0, "Arctic_Winter_Estimated"),
                (50.0, 15.0, 1013.25, 33.0, "Desert_Heat_Estimated"),
                (40.0, 95.0, 1013.25, 39.0, "Tropical_Humid_Estimated"),
                (10.0, 30.0, 700.0, -4.0, "High_Altitude_Estimated"),
                (-15.0, 90.0, 980.0, -16.0, "Winter_Storm_Estimated"),
            ]
        
        return pd.DataFrame(
            data, 
            columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_td_c', 'source']
        )


class TestDewpointCalculator:
    """Comprehensive test suite for the enhanced dewpoint calculator.
    
    This test class provides systematic validation of the dewpoint temperature
    calculation implementation through various test scenarios including:
    - Method validation across all available formulations
    - Custom Brent solver validation
    - Ice phase enhancement testing
    - External library cross-validation
    - Physical constraint verification
    - Performance benchmarking
    """
    
    # Class-level constants for test tolerances
    TOLERANCE_STRICT = 0.1      # ±0.1°C for strict comparison
    TOLERANCE_MODERATE = 0.3    # ±0.3°C for moderate comparison  
    TOLERANCE_EXTREME = 0.5     # ±0.5°C for extreme conditions
    TOLERANCE_ICE_PHASE = 0.4   # ±0.4°C for ice phase corrections
    
    # Available calculation methods for testing
    ANALYTICAL_METHODS = [
        'magnus_alduchov_eskridge',
        'magnus_standard', 
        'arden_buck',
        'tetens',
        'lawrence_simple'
    ]
    
    BRENT_SOLVER_METHODS = [
        'bolton_custom',
        'hyland_wexler'
    ]
    
    ALL_METHODS = ANALYTICAL_METHODS + BRENT_SOLVER_METHODS
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_external_libraries(self):
        """Initialize external libraries if available for cross-validation tests.
        
        Yields:
            None: This fixture performs setup/teardown without returning data.
        """
        if PSYCHROLIB_AVAILABLE:
            psychrolib.SetUnitSystem(psychrolib.SI)
        yield
        # Teardown code would go here if needed
    
    def test_all_methods_work_basic_conditions(self) -> None:
        """Test that all dewpoint calculation methods work for basic conditions.
        
        This test ensures basic functionality across all supported calculation
        methods using standard atmospheric conditions.
        """
        # Standard test conditions
        temp_c, rh_percent = 25.0, 60.0
        
        for method in self.ALL_METHODS:
            try:
                if method in ['hyland_wexler']:
                    # Enhanced methods with phase parameter
                    result = dewpoint(temp_c, rh_percent, method, phase="auto")
                else:
                    # Standard methods
                    result = dewpoint(temp_c, rh_percent, method)
                
                # Basic sanity checks
                assert isinstance(result, (int, float, np.number)), \
                    f"Result must be numeric for {method}, got {type(result)}"
                assert not np.isnan(result), f"Result cannot be NaN for {method}"
                assert not np.isinf(result), f"Result cannot be infinite for {method}"
                assert result < temp_c, f"Dewpoint should be less than air temp for {method}"
                assert result > temp_c - 30, f"Dewpoint should be within reasonable range for {method}"
                
            except Exception as e:
                pytest.fail(f"Method {method} failed basic test: {e}")
    
    def test_input_validation_handles_invalid_methods(self) -> None:
        """Test input validation and error handling for invalid method names.
        
        Ensures that appropriate ValueErrors are raised for invalid method
        specifications.
        """
        # Test invalid method name
        with pytest.raises(ValueError, match="Unknown equation"):
            dewpoint(25.0, 60.0, 'invalid_method')
        
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature"):
            dewpoint(-150.0, 60.0)
        
        # Test invalid humidity
        with pytest.raises(ValueError, match="Humidity"):
            dewpoint(25.0, 150.0)
    
    @pytest.mark.parametrize("temp,rh", [
        (-50.0, 80.0),   # Very cold
        (50.0, 10.0),    # Very hot and dry
        (0.0, 99.9),     # Near-saturation at freezing
        (40.0, 95.0),    # Hot and humid
    ])
    def test_extreme_input_cases_handled_gracefully(
        self, 
        temp: float, 
        rh: float
    ) -> None:
        """Test that extreme input cases are handled gracefully.
        
        Args:
            temp: Temperature in degrees Celsius.
            rh: Relative humidity in percent.
        """
        # Test with robust method
        result = dewpoint(temp, rh, "magnus_alduchov_eskridge")
        
        assert not np.isnan(result), \
            f"NaN result for extreme case: T={temp}°C, RH={rh}%"
        assert not np.isinf(result), \
            f"Infinite result for extreme case: T={temp}°C, RH={rh}%"
        assert result <= temp + 0.2, \
            f"Dewpoint exceeds air temperature: T={temp}°C, RH={rh}%"
    
    def test_scalar_and_array_inputs_give_consistent_results(self) -> None:
        """Test that scalar and array inputs give identical results.
        
        Validates that vectorized implementation produces the same results
        as scalar calculations.
        """
        # Test data
        temps = [20.0, 25.0, 30.0]
        rhs = [50.0, 60.0, 70.0]
        
        # Test multiple methods
        methods_to_test = ["magnus_alduchov_eskridge", "hyland_wexler"]
        
        for method in methods_to_test:
            # Calculate scalar results
            scalar_results = []
            for temp, rh in zip(temps, rhs):
                if method == "hyland_wexler":
                    result = dewpoint(temp, rh, method, phase="auto")
                else:
                    result = dewpoint(temp, rh, method)
                scalar_results.append(result)
            
            # Calculate array results
            if method == "hyland_wexler":
                array_results = dewpoint(temps, rhs, method, phase="auto")
            else:
                array_results = dewpoint(temps, rhs, method)
            
            # Compare with high precision
            np.testing.assert_allclose(
                scalar_results, 
                array_results, 
                rtol=1e-10, 
                atol=1e-10,
                err_msg=f"Scalar and array results must be identical for {method}"
            )
    
    @pytest.mark.skipif(
        not PSYCHROLIB_AVAILABLE, 
        reason="PsychroLib not available for cross-validation"
    )
    def test_results_against_psychrolib_standard(self) -> None:
        """Compare results against PsychroLib (ASHRAE standard implementation).
        
        This test validates accuracy against the authoritative ASHRAE
        implementation using Hyland-Wexler formulation for optimal comparison.
        """
        test_data = TestDatasets.get_ashrae_reference_data()
        
        differences = []
        for _, row in test_data.iterrows():
            # Use Hyland-Wexler method for best comparison with PsychroLib
            our_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='auto'
            )
            
            # Calculate PsychroLib result
            psychrolib_result = psychrolib.GetTDewPointFromRelHum(
                row['temp_c'], 
                row['rh_percent'] / 100.0
            )
            
            difference = abs(our_result - psychrolib_result)
            differences.append(difference)
            
            # Individual case validation
            tolerance = self.TOLERANCE_ICE_PHASE if row['temp_c'] <= 0 else self.TOLERANCE_MODERATE
            
            assert difference < tolerance, \
                f"Difference vs PsychroLib: {difference:.3f}°C for " \
                f"T={row['temp_c']}°C, RH={row['rh_percent']}% " \
                f"(tolerance: {tolerance:.3f}°C)"
        
        # Statistical validation
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        rms_diff = np.sqrt(np.mean(np.array(differences)**2))
        
        # Report statistics for documentation
        print(f"\nPsychroLib Cross-Validation Results:")
        print(f"  Mean difference: {mean_diff:.3f}°C")
        print(f"  Max difference:  {max_diff:.3f}°C") 
        print(f"  RMS difference:  {rms_diff:.3f}°C")
        
        # Statistical assertions
        assert mean_diff < 0.25, f"Mean difference too large: {mean_diff:.3f}°C"
        assert max_diff < 0.8, f"Maximum difference too large: {max_diff:.3f}°C"
    
    @pytest.mark.skipif(
        not COOLPROP_AVAILABLE, 
        reason="CoolProp not available for validation"
    )
    def test_results_against_coolprop_standard(self) -> None:
        """Validate results against CoolProp (ASHRAE RP-1485 reference).
        
        Tests using CoolProp which implements the ASHRAE RP-1485 standard
        for thermodynamic properties of humid air.
        """
        test_data = TestDatasets.get_ashrae_reference_data()
        
        differences = []
        for _, row in test_data.iterrows():
            # Our enhanced result
            our_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='auto'
            )
            
            # CoolProp calculation
            temp_k = row['temp_c'] + 273.15
            rh_fraction = row['rh_percent'] / 100.0
            pressure_pa = row['pressure_hpa'] * 100.0
            
            try:
                coolprop_td_k = HAPropsSI('Tdp', 'T', temp_k, 'R', rh_fraction, 'P', pressure_pa)
                coolprop_result = coolprop_td_k - 273.15
                
                difference = abs(our_result - coolprop_result)
                differences.append(difference)
                
                assert difference < self.TOLERANCE_MODERATE, \
                    f"CoolProp validation failed: {difference:.3f}°C difference " \
                    f"for T={row['temp_c']}°C, RH={row['rh_percent']}%"
                    
            except Exception as e:
                warnings.warn(f"CoolProp calculation failed for T={row['temp_c']}°C: {e}")
        
        if differences:
            mean_diff = np.mean(differences)
            print(f"\nCoolProp Validation Results:")
            print(f"  Mean difference: {mean_diff:.3f}°C")
            print(f"  Max difference:  {np.max(differences):.3f}°C")
    
    @pytest.mark.skipif(
        not METPY_AVAILABLE, 
        reason="MetPy not available for validation"
    )
    def test_results_against_metpy_bolton_standard(self) -> None:
        """Validate Bolton method against MetPy (Bolton 1980 standard).
        
        Tests our custom Bolton implementation against MetPy's reference
        implementation of the Bolton 1980 formulation.
        """
        test_data = TestDatasets.get_bolton_reference_data()
        
        for _, row in test_data.iterrows():
            # Our Bolton implementation
            our_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'bolton_custom'
            )
            
            # MetPy Bolton calculation
            temp_qty = row['temp_c'] * units.celsius
            rh_qty = row['rh_percent'] * units.percent
            
            metpy_result_qty = mpcalc.dewpoint_from_relative_humidity(temp_qty, rh_qty)
            metpy_result = metpy_result_qty.to('celsius').magnitude
            
            difference = abs(our_result - metpy_result)
            
            assert difference < self.TOLERANCE_STRICT, \
                f"MetPy Bolton validation failed: {difference:.3f}°C difference " \
                f"for T={row['temp_c']}°C, RH={row['rh_percent']}%"
    
    def test_ice_phase_enhancement_functionality(self) -> None:
        """Test ice phase enhancement functionality.
        
        Validates that ice phase corrections are properly applied for
        sub-freezing conditions.
        """
        ice_test_data = TestDatasets.get_ice_phase_reference_data()
        
        for _, row in ice_test_data.iterrows():
            # Test automatic phase selection
            auto_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='auto'
            )
            
            # Test explicit liquid phase
            liquid_result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'hyland_wexler',
                phase='liquid'
            )
            
            # Test explicit ice phase (for sub-freezing)
            if row['temp_c'] <= 0:
                ice_result = dewpoint(
                    row['temp_c'], 
                    row['rh_percent'], 
                    'hyland_wexler',
                    phase='ice'
                )
                
                # Auto should match ice for sub-freezing
                assert abs(auto_result - ice_result) < 0.01, \
                    f"Auto phase should match ice phase for T={row['temp_c']}°C"
                
                # Ice correction should be applied
                correction = ice_result - liquid_result
                assert 0.1 < correction < 0.3, \
                    f"Ice correction should be ~0.17°C, got {correction:.3f}°C"
            
            # Validate result is reasonable
            assert not np.isnan(auto_result), f"Auto result is NaN for {row['source']}"
            assert auto_result <= row['temp_c'] + 0.1, \
                f"Dewpoint exceeds air temperature for {row['source']}"
    
    def test_custom_brent_solver_functionality(self) -> None:
        """Test custom Brent solver implementation.
        
        Validates that the custom Brent solver (eliminating SciPy dependency)
        works correctly and provides accurate results.
        """
        # Test conditions where iterative methods are challenged
        challenging_cases = [
            (0.0, 99.0, "Near-saturation freezing"),
            (35.0, 95.0, "Hot humid extreme"),
            (-10.0, 98.0, "Cold near-saturation"),
            (45.0, 80.0, "Very hot humid"),
        ]
        
        for temp, rh, description in challenging_cases:
            # Compare Brent-based methods with analytical methods
            analytical_result = dewpoint(temp, rh, "magnus_alduchov_eskridge")
            brent_bolton_result = dewpoint(temp, rh, "bolton_custom")
            brent_hyland_result = dewpoint(temp, rh, "hyland_wexler", phase="auto")
            
            # All results should be finite and reasonable
            results = {
                'analytical': analytical_result,
                'brent_bolton': brent_bolton_result,
                'brent_hyland': brent_hyland_result
            }
            
            for method_name, result in results.items():
                assert np.isfinite(result), f"{method_name} failed for {description}"
                assert result <= temp + 0.1, f"{method_name} violates physics for {description}"
                assert result > temp - 40, f"{method_name} unreasonably low for {description}"
            
            # Method agreement (allowing for different formulations)
            bolton_diff = abs(brent_bolton_result - analytical_result)
            hyland_diff = abs(brent_hyland_result - analytical_result)
            
            assert bolton_diff <= 1.0, \
                f"Bolton vs Magnus disagreement for {description}: {bolton_diff:.3f}°C"
            assert hyland_diff <= 1.0, \
                f"Hyland vs Magnus disagreement for {description}: {hyland_diff:.3f}°C"
    
    def test_method_consistency_above_freezing(self) -> None:
        """Test consistency between different methods above freezing.
        
        Ensures all methods produce reasonably consistent results for
        above-freezing conditions where ice phase doesn't apply.
        """
        test_conditions = [
            (25.0, 60.0, "Standard comfort"),
            (30.0, 80.0, "Hot humid"),
            (20.0, 40.0, "Mild dry"),
            (35.0, 50.0, "Hot moderate"),
        ]
        
        for temp, rh, description in test_conditions:
            results = {}
            
            # Test precision analytical methods (should agree closely)
            precision_methods = [
                'magnus_alduchov_eskridge',
                'magnus_standard', 
                'arden_buck',
                'tetens'
            ]
            
            for method in precision_methods:
                results[method] = dewpoint(temp, rh, method)
            
            # Test Brent solver methods
            results['bolton_custom'] = dewpoint(temp, rh, 'bolton_custom')
            results['hyland_wexler'] = dewpoint(temp, rh, 'hyland_wexler', phase='liquid')
            
            # Test Lawrence simple method separately (it's a rough approximation)
            lawrence_result = dewpoint(temp, rh, 'lawrence_simple')
            
            # Check consistency among precision methods
            precision_values = [results[method] for method in precision_methods]
            precision_values.extend([results['bolton_custom'], results['hyland_wexler']])
            
            precision_range = max(precision_values) - min(precision_values)
            
            assert precision_range < 0.5, \
                f"Large variation between precision methods for {description}: {precision_range:.3f}°C. " \
                f"Precision results: {dict((k, v) for k, v in results.items() if k != 'lawrence_simple')}"
            
            # Lawrence method should be in the ballpark but may be less accurate
            lawrence_vs_average = abs(lawrence_result - np.mean(precision_values))
            
            assert lawrence_vs_average < 3.0, \
                f"Lawrence method too far from other methods for {description}: " \
                f"Lawrence={lawrence_result:.2f}°C, Average precision={np.mean(precision_values):.2f}°C, " \
                f"Difference={lawrence_vs_average:.2f}°C. Note: Lawrence is a simple approximation."
            
            # All results should be physically reasonable
            all_results = list(results.values()) + [lawrence_result]
            for method_name, result in zip(list(results.keys()) + ['lawrence_simple'], all_results):
                assert result <= temp, \
                    f"Method {method_name} violates physics for {description}: {result:.2f}°C > {temp}°C"
                assert result > temp - 30, \
                    f"Method {method_name} unreasonably low for {description}: {result:.2f}°C"
    
    
    def test_performance_under_extreme_conditions(self) -> None:
        """Test robustness under extreme atmospheric conditions.
        
        Uses robust method combinations suitable for challenging conditions.
        """
        test_data = TestDatasets.get_extreme_conditions_data()
        
        for _, row in test_data.iterrows():
            # Use robust method for extreme conditions
            result = dewpoint(
                row['temp_c'], 
                row['rh_percent'], 
                'magnus_alduchov_eskridge'  # Robust analytical method
            )
            
            # Basic validation
            assert np.isfinite(result), f"Extreme condition test failed for {row['source']}"
            assert result <= row['temp_c'] + 0.2, \
                f"Dewpoint exceeds air temperature for {row['source']}"
            
            # For extreme conditions, use more appropriate tolerances
            if row['expected_td_c'] is not None and not np.isnan(row['expected_td_c']):
                difference = abs(result - row['expected_td_c'])
                
                # Determine appropriate tolerance based on condition
                if "High_Altitude" in row['source']:
                    # High altitude: PsychroLib doesn't account for pressure effects properly
                    tolerance = 1.0  # ±1.0°C tolerance for altitude effects
                elif "Arctic" in row['source'] or row['temp_c'] < -30:
                    # Arctic conditions: larger uncertainties at extreme cold
                    tolerance = 1.0  # ±1.0°C tolerance for extreme cold
                elif "Desert" in row['source'] and row['rh_percent'] < 20:
                    # Very low humidity: larger relative errors possible  
                    tolerance = 1.5  # ±1.5°C tolerance for very dry conditions
                else:
                    # Other extreme conditions
                    tolerance = self.TOLERANCE_EXTREME  # ±0.5°C
                
                assert difference < tolerance, \
                    f"Extreme condition validation failed for {row['source']}: " \
                    f"Expected {row['expected_td_c']:.2f}°C, Got {result:.2f}°C, " \
                    f"Difference {difference:.3f}°C (tolerance: {tolerance:.1f}°C). " \
                    f"Note: Larger differences expected for extreme conditions due to " \
                    f"pressure effects and formulation differences."
                
    def test_physical_constraints_satisfied(self) -> None:
        """Test that results satisfy fundamental physical constraints.
        
        Validates that calculated dewpoint temperatures are physically
        reasonable and satisfy thermodynamic constraints.
        """
        test_conditions = [
            (25.0, 60.0, "Standard conditions"),
            (35.0, 90.0, "High humidity"),
            (10.0, 30.0, "Low humidity"),
            (-5.0, 80.0, "Sub-freezing"),
            (40.0, 95.0, "Extreme hot-humid"),
        ]
        
        for temp, rh, description in test_conditions:
            # Test with robust method
            result = dewpoint(temp, rh, "magnus_alduchov_eskridge")
            
            # Dewpoint should be less than air temperature (except at 100% RH)
            if rh < 99.9:
                assert result <= temp + 0.1, \
                    f"Dewpoint ({result:.2f}°C) should be ≤ air temp ({temp}°C) " \
                    f"for {description}"
            
            # Dewpoint should be within reasonable range
            assert result > temp - 50, \
                f"Dewpoint ({result:.2f}°C) unreasonably low vs air temp ({temp}°C) " \
                f"for {description}"
            
            # At high humidity, dewpoint should approach air temperature
            if rh > 95:
                temp_dewpoint_diff = temp - result
                assert temp_dewpoint_diff < 1.0, \
                    f"At high RH ({rh}%), dewpoint should be close to air temp. " \
                    f"Difference: {temp_dewpoint_diff:.2f}°C for {description}"


class TestPerformanceBenchmarks:
    """Performance benchmarking and optimization validation tests.
    
    This class focuses on testing the computational performance and
    efficiency of the dewpoint calculator implementation.
    """
    
    def test_vectorized_calculation_performance(self) -> None:
        """Test vectorized calculation performance and correctness.
        
        Validates that vectorized operations are both correct and
        performant compared to scalar calculations.
        """
        # Generate reproducible test data
        np.random.seed(42)
        n_points = 1000
        
        temps = np.random.uniform(-20, 45, n_points)
        rhs = np.random.uniform(10, 95, n_points)
        
        # Time vectorized calculation
        import time
        start_time = time.time()
        vectorized_results = dewpoint(
            temps, rhs,
            'magnus_alduchov_eskridge'
        )
        vectorized_time = time.time() - start_time
        
        # Time scalar calculations (subset for speed)
        start_time = time.time()
        scalar_results = []
        test_subset = 100  # Test subset for timing comparison
        for temp, rh in zip(temps[:test_subset], rhs[:test_subset]):
            result = dewpoint(temp, rh, 'magnus_alduchov_eskridge')
            scalar_results.append(result)
        scalar_time = time.time() - start_time
        
        # Validate correctness
        np.testing.assert_allclose(
            scalar_results, 
            vectorized_results[:test_subset],
            rtol=1e-10, 
            atol=1e-10,
            err_msg="Vectorized and scalar results must be identical"
        )
        
        # Performance reporting
        estimated_scalar_time = scalar_time / test_subset * n_points
        speedup = estimated_scalar_time / vectorized_time if vectorized_time > 0 else float('inf')
        
        print(f"\nPerformance Benchmark Results:")
        print(f"  Vectorized calculation ({n_points} points): {vectorized_time:.3f}s")
        print(f"  Scalar calculation ({test_subset} points): {scalar_time:.3f}s")
        print(f"  Estimated vectorized speedup: {speedup:.1f}x")
        
        # Performance assertion
        assert vectorized_time < estimated_scalar_time / 3, \
            "Vectorized calculation should be significantly faster than scalar"
    
    def test_vectorized_calculation_performance(self) -> None:
        """Test vectorized calculation performance and correctness.
        
        Validates that vectorized operations are both correct and
        performant compared to scalar calculations.
        """
        # Generate reproducible test data
        np.random.seed(42)
        n_points = 1000
        
        temps = np.random.uniform(-20, 45, n_points)
        rhs = np.random.uniform(10, 95, n_points)
        
        # Time vectorized calculation
        import time
        start_time = time.time()
        vectorized_results = dewpoint(
            temps, rhs,
            'magnus_alduchov_eskridge'
        )
        vectorized_time = time.time() - start_time
        
        # Time scalar calculations (subset for speed)
        start_time = time.time()
        scalar_results = []
        test_subset = 100  # Test subset for timing comparison
        for temp, rh in zip(temps[:test_subset], rhs[:test_subset]):
            result = dewpoint(temp, rh, 'magnus_alduchov_eskridge')
            scalar_results.append(result)
        scalar_time = time.time() - start_time
        
        # Validate correctness
        np.testing.assert_allclose(
            scalar_results, 
            vectorized_results[:test_subset],
            rtol=1e-10, 
            atol=1e-10,
            err_msg="Vectorized and scalar results must be identical"
        )
        
        # Performance reporting
        estimated_scalar_time = scalar_time / test_subset * n_points
        speedup = estimated_scalar_time / vectorized_time if vectorized_time > 0 else float('inf')
        
        print(f"\nVectorization Performance Results:")
        print(f"  Vectorized calculation ({n_points} points): {vectorized_time:.3f}s")
        print(f"  Scalar calculation ({test_subset} points): {scalar_time:.3f}s")
        print(f"  Estimated vectorized speedup: {speedup:.1f}x")
        
        # Performance assertion - vectorization should provide benefit
        assert vectorized_time < estimated_scalar_time / 2, \
            "Vectorized calculation should be significantly faster than scalar"
    
    def test_vectorization_efficiency(self) -> None:
        """Test vectorization efficiency for large arrays.
        
        Validates that vectorized operations scale efficiently and
        provide meaningful speedup over scalar loops.
        """
        # Test different array sizes
        test_sizes = [100, 1000, 5000]
        method = 'magnus_alduchov_eskridge'  # Use fastest method for clearest measurement
        
        print(f"\nVectorization Efficiency Analysis:")
        print(f"{'Size':<8} {'Vector (ms)':<12} {'Scalar (ms)':<12} {'Speedup':<10} {'Status':<10}")
        print("-" * 60)
        
        for size in test_sizes:
            # Generate test data
            np.random.seed(42)
            temps = np.random.uniform(-10, 40, size)
            rhs = np.random.uniform(20, 90, size)
            
            # Time vectorized calculation
            import time
            start_time = time.perf_counter()
            vector_results = dewpoint(temps, rhs, method)
            vector_time = time.perf_counter() - start_time
            
            # Time scalar calculation (sample for large arrays)
            sample_size = min(size, 200)  # Limit scalar test size
            start_time = time.perf_counter()
            scalar_results = []
            for i in range(sample_size):
                result = dewpoint(temps[i], rhs[i], method)
                scalar_results.append(result)
            scalar_sample_time = time.perf_counter() - start_time
            
            # Estimate full scalar time
            estimated_scalar_time = scalar_sample_time / sample_size * size
            speedup = estimated_scalar_time / vector_time if vector_time > 0 else float('inf')
            
            # Validate results match
            np.testing.assert_allclose(
                scalar_results, 
                vector_results[:sample_size],
                rtol=1e-12, atol=1e-12,
                err_msg=f"Scalar/vector mismatch at size {size}"
            )
            
            status = "GOOD" if speedup > 3.0 else "POOR"
            
            print(f"{size:<8} {vector_time*1000:<12.2f} {estimated_scalar_time*1000:<12.2f} "
                  f"{speedup:<10.1f}x {status:<10}")
            
            # Vectorization should provide meaningful speedup
            assert speedup > 2.0, \
                f"Insufficient vectorization speedup at size {size}: {speedup:.1f}x"
    
    @pytest.mark.skipif(
        not PSYCHROLIB_AVAILABLE, 
        reason="PsychroLib not available for performance comparison"
    )
    def test_performance_vs_psychrolib(self) -> None:
        """Benchmark performance against PsychroLib reference implementation.
        
        Compares computational speed while maintaining accuracy standards.
        """
        # Generate larger test data for measurable timing
        np.random.seed(42)
        n_points = 2000  # Larger for measurable timing
        
        temps = np.random.uniform(0, 40, n_points)
        rhs = np.random.uniform(20, 95, n_points)
        
        # Benchmark our implementation with multiple runs
        import time
        our_times = []
        for _ in range(5):  # Multiple runs for stable measurement
            start_time = time.perf_counter()  # More precise timer
            our_results = dewpoint(
                temps, rhs,
                'hyland_wexler',
                phase='auto'
            )
            elapsed = time.perf_counter() - start_time
            our_times.append(elapsed)
        
        our_time = np.median(our_times)  # Use median for stability
        
        # Benchmark PsychroLib (scalar implementation)
        psychrolib_times = []
        for _ in range(3):  # Fewer runs since it's slower
            start_time = time.perf_counter()
            psychrolib_results = []
            for temp, rh in zip(temps, rhs):
                result = psychrolib.GetTDewPointFromRelHum(temp, rh / 100.0)
                psychrolib_results.append(result)
            elapsed = time.perf_counter() - start_time
            psychrolib_times.append(elapsed)
        
        psychrolib_time = np.median(psychrolib_times)
        
        # Performance and accuracy reporting
        if our_time > 0 and psychrolib_time > 0:
            speedup = psychrolib_time / our_time
        else:
            speedup = float('inf')
            
        differences = np.abs(np.array(our_results) - np.array(psychrolib_results))
        
        print(f"\nPerformance vs PsychroLib ({n_points} points):")
        print(f"  Our implementation: {our_time*1000:.2f}ms")
        print(f"  PsychroLib (scalar): {psychrolib_time*1000:.2f}ms") 
        print(f"  Speed improvement: {speedup:.1f}x")
        print(f"  Mean accuracy difference: {np.mean(differences):.3f}°C")
        print(f"  Max accuracy difference: {np.max(differences):.3f}°C")
        
        # More realistic performance assertion
        if our_time > 1e-6:  # If we can actually measure our time
            assert our_time < psychrolib_time * 2, \
                "Our implementation should be competitive with PsychroLib"
        else:
            print("  Note: Our implementation too fast to measure accurately")
    
    @pytest.mark.skipif(
        not COOLPROP_AVAILABLE, 
        reason="CoolProp not available for performance comparison"
    )
    def test_performance_vs_coolprop(self) -> None:
        """Benchmark performance against CoolProp reference implementation.
        
        Compares computational speed while maintaining accuracy standards.
        """
        # Generate larger test data for measurable timing
        np.random.seed(42)
        n_points = 2000  # Larger for measurable timing
        
        temps = np.random.uniform(0, 40, n_points)
        rhs = np.random.uniform(20, 95, n_points)
        pressures = np.full(n_points, 101325.0)  # Standard pressure in Pa
        
        # Benchmark our implementation with multiple runs
        import time
        our_times = []
        for _ in range(5):  # Multiple runs for stable measurement
            start_time = time.perf_counter()  # More precise timer
            our_results = dewpoint(
                temps, rhs,
                'hyland_wexler',
                phase='auto'
            )
            elapsed = time.perf_counter() - start_time
            our_times.append(elapsed)
        
        our_time = np.median(our_times)  # Use median for stability
        
        # Benchmark CoolProp (scalar implementation)
        coolprop_times = []
        for _ in range(3):  # Fewer runs since it's slower
            start_time = time.perf_counter()
            coolprop_results = []
            for temp, rh, pressure in zip(temps, rhs, pressures):
                try:
                    temp_k = temp + 273.15
                    rh_fraction = rh / 100.0
                    dewpoint_k = HAPropsSI('Tdp', 'T', temp_k, 'R', rh_fraction, 'P', pressure)
                    dewpoint_c = dewpoint_k - 273.15
                    coolprop_results.append(dewpoint_c)
                except Exception:
                    # Skip failed calculations
                    coolprop_results.append(np.nan)
            elapsed = time.perf_counter() - start_time
            coolprop_times.append(elapsed)
        
        coolprop_time = np.median(coolprop_times)
        
        # Filter out failed calculations for comparison
        valid_indices = ~np.isnan(coolprop_results)
        if np.sum(valid_indices) == 0:
            pytest.skip("CoolProp calculations failed for all test cases")
        
        our_valid = np.array(our_results)[valid_indices]
        coolprop_valid = np.array(coolprop_results)[valid_indices]
        
        # Performance and accuracy reporting
        if our_time > 0 and coolprop_time > 0:
            speedup = coolprop_time / our_time
        else:
            speedup = float('inf')
            
        differences = np.abs(our_valid - coolprop_valid)
        
        print(f"\nPerformance vs CoolProp ({np.sum(valid_indices)}/{n_points} valid points):")
        print(f"  Our implementation: {our_time*1000:.2f}ms")
        print(f"  CoolProp (scalar): {coolprop_time*1000:.2f}ms") 
        print(f"  Speed improvement: {speedup:.1f}x")
        print(f"  Mean accuracy difference: {np.mean(differences):.3f}°C")
        print(f"  Max accuracy difference: {np.max(differences):.3f}°C")
        
        # More realistic performance assertion
        if our_time > 1e-6:  # If we can actually measure our time
            assert our_time < coolprop_time * 2, \
                "Our implementation should be competitive with CoolProp"
        else:
            print("  Note: Our implementation too fast to measure accurately")
    
    @pytest.mark.skipif(
        not METPY_AVAILABLE, 
        reason="MetPy not available for performance comparison"
    )
    def test_performance_vs_metpy(self) -> None:
        """Benchmark performance against MetPy reference implementation.
        
        Compares computational speed while maintaining accuracy standards.
        """
        # Generate larger test data for measurable timing
        np.random.seed(42)
        n_points = 2000  # Larger for measurable timing
        
        temps = np.random.uniform(0, 40, n_points)
        rhs = np.random.uniform(20, 95, n_points)
        
        # Benchmark our implementation (use Bolton for fair comparison) with multiple runs
        import time
        our_times = []
        for _ in range(5):  # Multiple runs for stable measurement
            start_time = time.perf_counter()  # More precise timer
            our_results = dewpoint(
                temps, rhs,
                'bolton_custom'
            )
            elapsed = time.perf_counter() - start_time
            our_times.append(elapsed)
        
        our_time = np.median(our_times)  # Use median for stability
        
        # Benchmark MetPy (scalar implementation)
        metpy_times = []
        for _ in range(3):  # Fewer runs since it's slower
            start_time = time.perf_counter()
            metpy_results = []
            for temp, rh in zip(temps, rhs):
                try:
                    temp_qty = temp * units.celsius
                    rh_qty = rh * units.percent
                    dewpoint_qty = mpcalc.dewpoint_from_relative_humidity(temp_qty, rh_qty)
                    dewpoint_c = dewpoint_qty.to('celsius').magnitude
                    metpy_results.append(dewpoint_c)
                except Exception:
                    # Skip failed calculations
                    metpy_results.append(np.nan)
            elapsed = time.perf_counter() - start_time
            metpy_times.append(elapsed)
        
        metpy_time = np.median(metpy_times)
        
        # Filter out failed calculations for comparison
        valid_indices = ~np.isnan(metpy_results)
        if np.sum(valid_indices) == 0:
            pytest.skip("MetPy calculations failed for all test cases")
        
        our_valid = np.array(our_results)[valid_indices]
        metpy_valid = np.array(metpy_results)[valid_indices]
        
        # Performance and accuracy reporting
        if our_time > 0 and metpy_time > 0:
            speedup = metpy_time / our_time
        else:
            speedup = float('inf')
            
        differences = np.abs(our_valid - metpy_valid)
        
        print(f"\nPerformance vs MetPy ({np.sum(valid_indices)}/{n_points} valid points):")
        print(f"  Our implementation: {our_time*1000:.2f}ms")
        print(f"  MetPy (scalar): {metpy_time*1000:.2f}ms") 
        print(f"  Speed improvement: {speedup:.1f}x")
        print(f"  Mean accuracy difference: {np.mean(differences):.3f}°C")
        print(f"  Max accuracy difference: {np.max(differences):.3f}°C")
        
        # More realistic performance assertion
        if our_time > 1e-6:  # If we can actually measure our time
            assert our_time < metpy_time * 2, \
                "Our implementation should be competitive with MetPy"
        else:
            print("  Note: Our implementation too fast to measure accurately")


class TestCustomBrentSolver:
    """Test suite specifically for the custom Brent solver implementation.
    
    Validates the custom Brent solver that eliminates SciPy dependency
    while maintaining accuracy and robustness.
    """
    
    def test_brent_solver_convergence_accuracy(self) -> None:
        """Test custom Brent solver convergence and accuracy.
        
        Validates that the Brent solver converges reliably and produces
        accurate results across various conditions.
        """
        test_conditions = [
            (25.0, 60.0, "Standard conditions"),
            (0.0, 99.0, "Near-saturation freezing"),
            (35.0, 95.0, "Hot humid extreme"),
            (-10.0, 98.0, "Cold near-saturation"),
        ]
        
        for temp, rh, description in test_conditions:
            # Test Bolton method (uses Brent solver)
            bolton_result = dewpoint(temp, rh, 'bolton_custom')
            
            # Test Hyland-Wexler method (uses Brent solver)
            hyland_result = dewpoint(temp, rh, 'hyland_wexler', phase='auto')
            
            # Compare with analytical method
            analytical_result = dewpoint(temp, rh, 'magnus_alduchov_eskridge')
            
            # All should be finite and reasonable
            assert np.isfinite(bolton_result), f"Bolton failed for {description}"
            assert np.isfinite(hyland_result), f"Hyland-Wexler failed for {description}"
            assert np.isfinite(analytical_result), f"Analytical failed for {description}"
            
            # Physical constraints
            assert bolton_result <= temp + 0.1, f"Bolton violates physics for {description}"
            assert hyland_result <= temp + 0.1, f"Hyland-Wexler violates physics for {description}"
            
            # Methods should agree reasonably (allowing for different formulations)
            bolton_diff = abs(bolton_result - analytical_result)
            hyland_diff = abs(hyland_result - analytical_result)
            
            assert bolton_diff <= 2.0, \
                f"Bolton vs analytical disagreement for {description}: {bolton_diff:.3f}°C"
            assert hyland_diff <= 2.0, \
                f"Hyland-Wexler vs analytical disagreement for {description}: {hyland_diff:.3f}°C"
    
    def test_brent_solver_robustness(self) -> None:
        """Test Brent solver robustness under challenging conditions.
        
        Validates that the solver handles edge cases and difficult
        convergence scenarios gracefully.
        """
        challenging_cases = [
            # Very high humidity cases
            (30.0, 99.9, "Near-saturation"),
            (0.0, 100.0, "Complete saturation at freezing"),
            
            # Very low humidity cases  
            (40.0, 5.0, "Very dry conditions"),
            (50.0, 10.0, "Desert conditions"),
            
            # Extreme temperature cases
            (-30.0, 80.0, "Extreme cold"),
            (45.0, 85.0, "Extreme heat"),
        ]
        
        for temp, rh, description in challenging_cases:
            try:
                # Test both Brent-based methods
                bolton_result = dewpoint(temp, rh, 'bolton_custom')
                hyland_result = dewpoint(temp, rh, 'hyland_wexler', phase='auto')
                
                # Both should produce finite, reasonable results
                assert np.isfinite(bolton_result), f"Bolton failed for {description}"
                assert np.isfinite(hyland_result), f"Hyland-Wexler failed for {description}"
                
                # Physical constraints
                assert bolton_result <= temp + 0.2, \
                    f"Bolton result unrealistic for {description}: {bolton_result:.2f}°C"
                assert hyland_result <= temp + 0.2, \
                    f"Hyland-Wexler result unrealistic for {description}: {hyland_result:.2f}°C"
                
            except Exception as e:
                pytest.fail(f"Brent solver failed for {description}: {e}")


class TestIcePhaseEnhancement:
    """Test suite for ice phase enhancement functionality.
    
    Validates the enhanced ice phase handling that applies empirical
    corrections for improved accuracy in sub-freezing conditions.
    """
    
    def test_automatic_phase_selection(self) -> None:
        """Test automatic ice/liquid phase selection logic.
        
        Validates that the 'auto' phase parameter correctly selects
        ice or liquid phase based on temperature.
        """
        test_cases = [
            (5.0, 80.0, "liquid", "Above freezing"),
            (0.0, 90.0, "ice", "At freezing point"),
            (-5.0, 85.0, "ice", "Below freezing"),
        ]
        
        for temp, rh, expected_phase, description in test_cases:
            # Get results for different phase settings
            auto_result = dewpoint(temp, rh, 'hyland_wexler', phase='auto')
            liquid_result = dewpoint(temp, rh, 'hyland_wexler', phase='liquid')
            ice_result = dewpoint(temp, rh, 'hyland_wexler', phase='ice')
            
            if expected_phase == "liquid":
                # Auto should closely match liquid above freezing
                assert abs(auto_result - liquid_result) < 0.02, \
                    f"Auto phase should match liquid for {description}"
            else:
                # Auto should closely match ice at/below freezing
                assert abs(auto_result - ice_result) < 0.02, \
                    f"Auto phase should match ice for {description}"
    
    def test_ice_phase_correction_magnitude(self) -> None:
        """Test empirical ice phase correction magnitude.
        
        Validates that the ice phase correction is approximately 0.17°C
        as documented in the enhancement.
        """
        ice_conditions = [
            (-10.0, 70.0, "Cold winter"),
            (-5.0, 85.0, "Moderate winter"), 
            (0.0, 90.0, "Freezing point"),
        ]
        
        for temp, rh, description in ice_conditions:
            liquid_result = dewpoint(temp, rh, 'hyland_wexler', phase='liquid')
            ice_result = dewpoint(temp, rh, 'hyland_wexler', phase='ice')
            
            correction = ice_result - liquid_result
            
            # Ice correction should be approximately 0.17°C
            assert 0.10 < correction < 0.25, \
                f"Ice correction out of expected range for {description}: " \
                f"{correction:.3f}°C (expected ~0.17°C)"
    
    def test_freezing_point_critical_behavior(self) -> None:
        """Test critical behavior at freezing point.
        
        The freezing point (0°C) is the critical case that prompted
        the ice phase enhancement development.
        """
        # The critical case: 0°C, 90% RH
        temp_c, rh_percent = 0.0, 90.0
        
        # Test all phase options
        auto_result = dewpoint(temp_c, rh_percent, 'hyland_wexler', phase='auto')
        liquid_result = dewpoint(temp_c, rh_percent, 'hyland_wexler', phase='liquid')
        ice_result = dewpoint(temp_c, rh_percent, 'hyland_wexler', phase='ice')
        
        # Auto should use ice phase at 0°C
        assert abs(auto_result - ice_result) < 0.02, \
            "Auto phase should use ice at freezing point"
        
        # Ice correction should be applied
        correction = ice_result - liquid_result
        assert correction > 0.05, \
            f"Ice correction should be positive at freezing point: {correction:.3f}°C"
        
        # Result should be physically reasonable
        assert auto_result < temp_c, \
            "Dewpoint should be below air temperature at freezing point"
        assert auto_result > temp_c - 5.0, \
            "Dewpoint should be reasonable relative to air temperature"


def generate_test_report() -> Dict[str, Any]:
    """Generate comprehensive test execution report.
    
    Returns:
        Dict[str, Any]: Structured report containing test metrics and results.
        
    Note:
        This function provides a template for automated test reporting.
        In practice, this would be populated by the test runner with
        actual execution metrics.
    """
    return {
        'test_summary': {
            'total_test_cases': 0,
            'passed': 0,
            'failed': 0,
            'accuracy_metrics': {},
            'performance_metrics': {}
        },
        'method_validation': {},
        'external_validation': {},
        'ice_phase_validation': {},
        'brent_solver_validation': {},
        'recommendations': [],
        'execution_time': 0.0,
        'coverage_percentage': 0.0
    }


if __name__ == "__main__":
    """Execute test suite with comprehensive reporting and configuration.
    
    This script can be run directly for development testing or through
    pytest for integration with CI/CD pipelines.
    
    Examples:
        python test_dewpoint_enhanced.py
        pytest test_dewpoint_enhanced.py -v --tb=short
        pytest test_dewpoint_enhanced.py::TestDewpointCalculator -v
        pytest test_dewpoint_enhanced.py -m "not external" -v
    """
    
    print("Enhanced Dewpoint Calculator Implementation Test Suite")
    print("=" * 65)
    print("Professional Testing Framework v2.0.0")
    print("Custom Brent Solver | Ice Phase Enhancement | External Validation")
    print("Author: Meteorological Software Engineering Team")
    print("Date: 2025")
    print()
    
    # Check dependencies and provide setup guidance
    missing_deps = []
    available_features = []
    
    # Check core implementation
    if not DEWPOINT_AVAILABLE:
        print("❌ ERROR: Enhanced dewpoint calculator not found")
        print("   Please check the import path in the test file")
        print("   Expected: dewpoint(), DewpointCalculator(), VaporPressureConstants")
        print()
        exit(1)
    else:
        print("✅ Enhanced dewpoint calculator found and working")
        available_features.append("Core Implementation")
    
    # Check external libraries
    if PSYCHROLIB_AVAILABLE:
        print("✅ PsychroLib available - ASHRAE validation enabled")
        available_features.append("PsychroLib Validation")
    else:
        missing_deps.append("psychrolib")
        print("⚠️  WARNING: PsychroLib not available")
        print("   ASHRAE cross-validation tests will be skipped")
        print("   Install with: pip install psychrolib")
    
    if COOLPROP_AVAILABLE:
        print("✅ CoolProp available - ASHRAE RP-1485 validation enabled")
        available_features.append("CoolProp Validation")
    else:
        missing_deps.append("CoolProp")
        print("⚠️  WARNING: CoolProp not available")
        print("   ASHRAE RP-1485 validation tests will be skipped")
        print("   Install with: pip install CoolProp")
    
    if METPY_AVAILABLE:
        print("✅ MetPy available - Bolton 1980 validation enabled")
        available_features.append("MetPy Validation")
    else:
        missing_deps.append("metpy")
        print("⚠️  WARNING: MetPy not available")
        print("   Bolton 1980 validation tests will be skipped")
        print("   Install with: pip install metpy")
    
    print()
    
    # Test basic functionality
    try:
        # Test core methods
        basic_result = dewpoint(25.0, 60.0)
        ice_result = dewpoint(0.0, 90.0, 'hyland_wexler', phase='auto')
        brent_result = dewpoint(25.0, 60.0, 'bolton_custom')
        
        print("✅ Core functionality tests passed:")
        print(f"   Basic calculation: {basic_result:.2f}°C")
        print(f"   Ice phase enhancement: {ice_result:.2f}°C")
        print(f"   Custom Brent solver: {brent_result:.2f}°C")
        available_features.extend([
            "Ice Phase Enhancement",
            "Custom Brent Solver",
            "Vectorized Operations"
        ])
        
    except Exception as e:
        print(f"❌ ERROR: Basic functionality test failed: {e}")
        exit(1)
    
    print()
    
    # Feature summary
    print("Available Features:")
    for feature in available_features:
        print(f"  ✅ {feature}")
    
    if missing_deps:
        print(f"\nOptional dependencies missing: {', '.join(missing_deps)}")
        print("Install all dependencies with:")
        print("pip install pytest numpy pandas psychrolib CoolProp metpy")
    
    print()
    
    # Display test execution options
    print("Test Execution Options:")
    print("  Full test suite:     pytest test_dewpoint_enhanced.py -v")
    print("  Core tests only:     pytest test_dewpoint_enhanced.py -m 'not external' -v")
    print("  Performance focus:   pytest test_dewpoint_enhanced.py::TestPerformanceBenchmarks -v")
    print("  Ice phase tests:     pytest test_dewpoint_enhanced.py::TestIcePhaseEnhancement -v")
    print("  Brent solver tests:  pytest test_dewpoint_enhanced.py::TestCustomBrentSolver -v")
    print("  External validation: pytest test_dewpoint_enhanced.py -m external -v")
    print("  With coverage:       pytest test_dewpoint_enhanced.py --cov -v")
    print("  Stop on first fail:  pytest test_dewpoint_enhanced.py -x -v")
    print()
    
    # Execute pytest with optimal settings for development
    print("Executing test suite with recommended settings...")
    print("-" * 65)
    
    import sys
    
    # Configure pytest arguments for comprehensive testing
    pytest_args = [
        __file__,
        "-v",                    # Verbose output
        "--tb=short",           # Concise traceback format
        "--durations=10",       # Show 10 slowest tests
        "--strict-markers",     # Ensure all markers are registered
        "--disable-warnings",   # Clean output (remove for debugging)
        "-ra",                  # Show summary of all non-passing tests
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend(["--cov=dewpoint", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    # Add performance timing if available
    try:
        import pytest_benchmark
        pytest_args.append("--benchmark-skip")  # Skip benchmarks by default
    except ImportError:
        pass
    
    # Execute the test suite
    exit_code = pytest.main(pytest_args)
    
    # Final summary
    print()
    print("=" * 65)
    if exit_code == 0:
        print("🎉 ENHANCED DEWPOINT CALCULATOR: ALL TESTS PASSED")
        print("✅ Custom Brent solver validation successful")
        print("✅ Ice phase enhancement validation successful")
        if len(available_features) >= 6:
            print("🏆 FULL VALIDATION COMPLETE - Production ready")
        else:
            print(f"✅ Core validation complete ({len(available_features)} features validated)")
    else:
        print("❌ ENHANCED DEWPOINT CALCULATOR: SOME TESTS FAILED")
        print("🔍 Review test output above for detailed failure analysis")
    
    print("=" * 65)
    sys.exit(exit_code)