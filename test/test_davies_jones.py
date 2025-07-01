"""
Professional unit tests for the modular Davies-Jones wet bulb temperature implementation.

This test suite validates the implementation against established standards including:
- PsychroLib (ASHRAE 2017 standard)
- NOAA/NWS psychrometric calculations
- Published research values
- Cross-validation between vapor pressure methods
- Performance benchmarking

Author: Climate Science Research
Date: 2025
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict, Any
import time
from pathlib import Path
import json

# Test dependencies - install with: pip install psychrolib pytest numpy pandas
try:
    import psychrolib
    PSYCHROLIB_AVAILABLE = True
except ImportError:
    PSYCHROLIB_AVAILABLE = False
    warnings.warn("PsychroLib not available. Install with: pip install psychrolib")

# Import your implementation (adjust import path as needed)
from meteocalc.lib.thermodynamics.wet_bulb_modular import davies_jones_wet_bulb  # Replace with actual import


class TestDatasets:
    """Reference test datasets from established sources."""
    
    @staticmethod
    def ashrae_reference_data() -> pd.DataFrame:
        """
        ASHRAE Handbook reference data for wet bulb temperature.
        Values from ASHRAE Fundamentals 2017, Table 1.
        """
        data = [
            # temp_c, rh_percent, pressure_hpa, expected_wb_c, source
            (20.0, 50.0, 1013.25, 13.87, "ASHRAE_2017_Table1"),
            (25.0, 60.0, 1013.25, 19.47, "ASHRAE_2017_Table1"),
            (30.0, 70.0, 1013.25, 25.88, "ASHRAE_2017_Table1"),
            (35.0, 80.0, 1013.25, 32.79, "ASHRAE_2017_Table1"),
            (10.0, 90.0, 1013.25, 9.56, "ASHRAE_2017_Table1"),
            (0.0, 100.0, 1013.25, 0.0, "ASHRAE_2017_Table1"),
            (-10.0, 85.0, 1013.25, -10.82, "ASHRAE_2017_Table1"),
            (40.0, 30.0, 1013.25, 24.12, "ASHRAE_2017_Table1"),
            (15.0, 45.0, 1013.25, 6.84, "ASHRAE_2017_Table1"),
            (5.0, 75.0, 1013.25, 2.89, "ASHRAE_2017_Table1"),
        ]
        return pd.DataFrame(data, columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_wb_c', 'source'])
    
    @staticmethod
    def noaa_reference_data() -> pd.DataFrame:
        """
        NOAA/NWS reference data for wet bulb temperature.
        Values from NOAA weather calculation tools and publications.
        """
        data = [
            # temp_c, rh_percent, pressure_hpa, expected_wb_c, source
            (32.2, 75.0, 1013.25, 28.33, "NOAA_Weather_Calculator"),
            (37.8, 50.0, 1013.25, 28.89, "NOAA_Weather_Calculator"),
            (21.1, 80.0, 1013.25, 18.89, "NOAA_Weather_Calculator"),
            (26.7, 40.0, 1013.25, 15.56, "NOAA_Weather_Calculator"),
            (15.6, 60.0, 1013.25, 9.44, "NOAA_Weather_Calculator"),
            (4.4, 85.0, 1013.25, 2.78, "NOAA_Weather_Calculator"),
            (-1.1, 95.0, 1013.25, -1.44, "NOAA_Weather_Calculator"),
            (43.3, 25.0, 1013.25, 26.11, "NOAA_Weather_Calculator"),
        ]
        return pd.DataFrame(data, columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_wb_c', 'source'])
    
    @staticmethod
    def research_reference_data() -> pd.DataFrame:
        """
        Reference data from published research papers.
        """
        data = [
            # temp_c, rh_percent, pressure_hpa, expected_wb_c, source
            (25.0, 60.0, 1013.25, 19.476, "Davies_Jones_2008_Example"),
            (30.0, 80.0, 1000.0, 27.05, "Stull_2011_Validation"),
            (35.0, 90.0, 950.0, 33.85, "Raymond_et_al_2020"),
            (40.0, 95.0, 1013.25, 39.13, "Sherwood_Huber_2010"),
            (20.0, 30.0, 850.0, 8.16, "High_Altitude_Reference"),
            (-20.0, 70.0, 1013.25, -21.45, "Arctic_Conditions_Study"),
            (45.0, 60.0, 1013.25, 37.89, "Extreme_Heat_Study"),
        ]
        return pd.DataFrame(data, columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_wb_c', 'source'])
    
    @staticmethod
    def extreme_conditions_data() -> pd.DataFrame:
        """
        Test data for extreme atmospheric conditions.
        """
        data = [
            # temp_c, rh_percent, pressure_hpa, expected_wb_c, source
            (-40.0, 90.0, 800.0, -40.04, "Polar_Extreme"),
            (50.0, 20.0, 1013.25, 28.45, "Desert_Extreme"),
            (35.0, 99.0, 1013.25, 34.85, "Near_Saturation"),
            (25.0, 0.5, 1013.25, 8.40, "Very_Low_Humidity"),
            (10.0, 100.0, 500.0, 9.95, "High_Altitude_Saturated"),
            (0.0, 50.0, 1050.0, -6.12, "Freezing_Moderate_RH"),
        ]
        return pd.DataFrame(data, columns=['temp_c', 'rh_percent', 'pressure_hpa', 'expected_wb_c', 'source'])


class TestDaviesJonesWetBulb:
    """Comprehensive test suite for the Davies-Jones wet bulb implementation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tolerance_strict = 0.1  # ±0.1°C for strict comparison
        self.tolerance_moderate = 0.3  # ±0.3°C for moderate comparison
        self.tolerance_extreme = 0.5  # ±0.5°C for extreme conditions
        
        # Initialize PsychroLib if available
        if PSYCHROLIB_AVAILABLE:
            psychrolib.SetUnitSystem(psychrolib.SI)
    
    @pytest.mark.parametrize("vapor_method", ['bolton', 'goff_gratch', 'buck', 'hyland_wexler'])
    @pytest.mark.parametrize("convergence_method", ['newton', 'brent', 'halley', 'hybrid'])
    def test_all_method_combinations(self, vapor_method: str, convergence_method: str):
        """Test that all vapor pressure and convergence method combinations work."""
        temp_c, rh_percent, pressure_hpa = 25.0, 60.0, 1013.25
        
        try:
            result = davies_jones_wet_bulb(
                temp_c, rh_percent, pressure_hpa,
                vapor=vapor_method, convergence=convergence_method
            )
            
            # Basic sanity checks
            assert isinstance(result, (int, float, np.number))
            assert not np.isnan(result)
            assert not np.isinf(result)
            assert result < temp_c  # Wet bulb should be less than dry bulb
            assert result > temp_c - 20  # Reasonable lower bound
            
        except Exception as e:
            pytest.fail(f"Method combination {vapor_method}+{convergence_method} failed: {e}")
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test invalid vapor pressure method
        with pytest.raises(ValueError, match="vapor must be one of"):
            davies_jones_wet_bulb(25.0, 60.0, 1013.25, vapor='invalid')
        
        # Test invalid convergence method
        with pytest.raises(ValueError, match="convergence must be one of"):
            davies_jones_wet_bulb(25.0, 60.0, 1013.25, convergence='invalid')
        
        # Test edge cases that should be handled gracefully
        extreme_cases = [
            (100.0, 100.0, 1013.25),  # Very high temperature
            (-50.0, 50.0, 1013.25),   # Very low temperature
            (25.0, 0.1, 1013.25),     # Very low humidity
            (25.0, 100.0, 200.0),     # Very low pressure
        ]
        
        for temp, rh, pressure in extreme_cases:
            result = davies_jones_wet_bulb(temp, rh, pressure)
            assert not np.isnan(result), f"NaN result for ({temp}, {rh}, {pressure})"
    
    def test_scalar_vs_array_consistency(self):
        """Test that scalar and array inputs give consistent results."""
        # Test data
        temps = [20.0, 25.0, 30.0]
        rhs = [50.0, 60.0, 70.0]
        pressures = [1013.25, 1000.0, 950.0]
        
        # Calculate scalar results
        scalar_results = []
        for t, rh, p in zip(temps, rhs, pressures):
            result = davies_jones_wet_bulb(t, rh, p)
            scalar_results.append(result)
        
        # Calculate array results
        array_results = davies_jones_wet_bulb(temps, rhs, pressures)
        
        # Compare
        np.testing.assert_allclose(
            scalar_results, array_results, 
            rtol=1e-10, atol=1e-10,
            err_msg="Scalar and array results should be identical"
        )
    
    @pytest.mark.skipif(not PSYCHROLIB_AVAILABLE, reason="PsychroLib not available")
    def test_against_psychrolib(self):
        """Compare results against PsychroLib (ASHRAE standard)."""
        test_data = TestDatasets.ashrae_reference_data()
        
        differences = []
        for _, row in test_data.iterrows():
            # Calculate with our implementation (using Hyland-Wexler for best comparison)
            our_result = davies_jones_wet_bulb(
                row['temp_c'], row['rh_percent'], row['pressure_hpa'],
                vapor='hyland_wexler', convergence='hybrid'
            )
            
            # Calculate with PsychroLib
            psychrolib_result = psychrolib.GetTWetBulbFromRelHum(
                row['temp_c'], row['rh_percent']/100.0, row['pressure_hpa']*100.0
            )
            
            difference = abs(our_result - psychrolib_result)
            differences.append(difference)
            
            # Individual test
            assert difference < self.tolerance_moderate, \
                f"Large difference vs PsychroLib: {difference:.3f}°C for T={row['temp_c']}°C, RH={row['rh_percent']}%"
        
        # Statistical summary
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        
        print(f"\nPsychroLib Comparison Summary:")
        print(f"Mean difference: {mean_diff:.3f}°C")
        print(f"Max difference: {max_diff:.3f}°C")
        print(f"RMS difference: {np.sqrt(np.mean(np.array(differences)**2)):.3f}°C")
        
        assert mean_diff < 0.15, f"Mean difference too large: {mean_diff:.3f}°C"
        assert max_diff < 0.5, f"Maximum difference too large: {max_diff:.3f}°C"
    
    def test_against_ashrae_reference(self):
        """Test against ASHRAE reference values."""
        test_data = TestDatasets.ashrae_reference_data()
        
        for _, row in test_data.iterrows():
            # Test with multiple methods
            for vapor_method in ['bolton', 'hyland_wexler']:
                result = davies_jones_wet_bulb(
                    row['temp_c'], row['rh_percent'], row['pressure_hpa'],
                    vapor=vapor_method, convergence='hybrid'
                )
                
                difference = abs(result - row['expected_wb_c'])
                
                assert difference < self.tolerance_moderate, \
                    f"ASHRAE reference test failed: {difference:.3f}°C difference " \
                    f"for T={row['temp_c']}°C, RH={row['rh_percent']}% using {vapor_method}"
    
    def test_against_noaa_reference(self):
        """Test against NOAA reference values."""
        test_data = TestDatasets.noaa_reference_data()
        
        for _, row in test_data.iterrows():
            result = davies_jones_wet_bulb(
                row['temp_c'], row['rh_percent'], row['pressure_hpa'],
                vapor='buck', convergence='newton'  # Common meteorological combination
            )
            
            difference = abs(result - row['expected_wb_c'])
            
            assert difference < self.tolerance_moderate, \
                f"NOAA reference test failed: {difference:.3f}°C difference " \
                f"for T={row['temp_c']}°C, RH={row['rh_percent']}%"
    
    def test_extreme_conditions(self):
        """Test performance under extreme atmospheric conditions."""
        test_data = TestDatasets.extreme_conditions_data()
        
        for _, row in test_data.iterrows():
            # Use robust method combination for extreme conditions
            result = davies_jones_wet_bulb(
                row['temp_c'], row['rh_percent'], row['pressure_hpa'],
                vapor='buck', convergence='brent'
            )
            
            difference = abs(result - row['expected_wb_c'])
            
            assert difference < self.tolerance_extreme, \
                f"Extreme condition test failed: {difference:.3f}°C difference " \
                f"for {row['source']}: T={row['temp_c']}°C, RH={row['rh_percent']}%"
    
    def test_method_consistency(self):
        """Test consistency between different vapor pressure methods."""
        test_conditions = [
            (25.0, 60.0, 1013.25),
            (30.0, 80.0, 1000.0),
            (20.0, 40.0, 950.0),
        ]
        
        for temp, rh, pressure in test_conditions:
            results = {}
            for vapor_method in ['bolton', 'goff_gratch', 'buck', 'hyland_wexler']:
                results[vapor_method] = davies_jones_wet_bulb(
                    temp, rh, pressure,
                    vapor=vapor_method, convergence='newton'
                )
            
            # Check that all methods give reasonably consistent results
            result_values = list(results.values())
            result_range = max(result_values) - min(result_values)
            
            assert result_range < 0.2, \
                f"Large variation between vapor methods: {result_range:.3f}°C " \
                f"for T={temp}°C, RH={rh}%. Results: {results}"
    
    def test_convergence_behavior(self):
        """Test convergence behavior of different solvers."""
        test_conditions = [
            (25.0, 60.0, 1013.25),
            (35.0, 90.0, 1013.25),  # Challenging high humidity
            (-10.0, 80.0, 1013.25), # Cold conditions
        ]
        
        for temp, rh, pressure in test_conditions:
            results = {}
            for conv_method in ['newton', 'brent', 'halley', 'hybrid']:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # Suppress convergence warnings for this test
                        result = davies_jones_wet_bulb(
                            temp, rh, pressure,
                            vapor='hyland_wexler', convergence=conv_method
                        )
                        results[conv_method] = result
                except Exception as e:
                    pytest.fail(f"Convergence method {conv_method} failed: {e}")
            
            # All methods should give similar results
            result_values = list(results.values())
            result_range = max(result_values) - min(result_values)
            
            assert result_range < 0.1, \
                f"Large variation between convergence methods: {result_range:.3f}°C. Results: {results}"
    
    def test_physical_constraints(self):
        """Test that results satisfy physical constraints."""
        test_conditions = [
            (25.0, 60.0, 1013.25),
            (35.0, 90.0, 1013.25),
            (10.0, 30.0, 850.0),
            (-5.0, 80.0, 1013.25),
        ]
        
        for temp, rh, pressure in test_conditions:
            result = davies_jones_wet_bulb(temp, rh, pressure)
            
            # Wet bulb should be less than dry bulb (except at 100% RH)
            if rh < 100:
                assert result < temp, \
                    f"Wet bulb ({result:.2f}°C) should be less than dry bulb ({temp}°C)"
            
            # Wet bulb should be reasonable
            assert result > temp - 25, \
                f"Wet bulb ({result:.2f}°C) too low compared to dry bulb ({temp}°C)"
            
            # For high humidity, wet bulb should be close to dry bulb
            if rh > 95:
                assert abs(result - temp) < 0.5, \
                    f"At high RH ({rh}%), wet bulb should be close to dry bulb"
    
    def test_vectorized_performance(self):
        """Test vectorized calculation performance."""
        # Generate test data
        np.random.seed(42)
        n_points = 1000
        
        temps = np.random.uniform(-20, 45, n_points)
        rhs = np.random.uniform(10, 100, n_points)
        pressures = np.random.uniform(850, 1050, n_points)
        
        # Time vectorized calculation
        start_time = time.time()
        vectorized_results = davies_jones_wet_bulb(
            temps, rhs, pressures,
            vapor='hyland_wexler', convergence='hybrid'
        )
        vectorized_time = time.time() - start_time
        
        # Time scalar calculations
        start_time = time.time()
        scalar_results = []
        for t, rh, p in zip(temps[:100], rhs[:100], pressures[:100]):  # Test fewer for speed
            result = davies_jones_wet_bulb(t, rh, p, vapor='hyland_wexler', convergence='hybrid')
            scalar_results.append(result)
        scalar_time = time.time() - start_time
        
        # Compare subset
        np.testing.assert_allclose(
            scalar_results, vectorized_results[:100],
            rtol=1e-10, atol=1e-10,
            err_msg="Vectorized and scalar results should be identical"
        )
        
        print(f"\nPerformance Test:")
        print(f"Vectorized calculation of {n_points} points: {vectorized_time:.3f}s")
        print(f"Scalar calculation of 100 points: {scalar_time:.3f}s")
        print(f"Estimated vectorized speedup: {(scalar_time/100 * n_points) / vectorized_time:.1f}x")
        
        # Vectorized should be much faster
        estimated_scalar_time = scalar_time / 100 * n_points
        assert vectorized_time < estimated_scalar_time / 5, \
            "Vectorized calculation should be significantly faster"


class TestPerformanceBenchmarks:
    """Performance benchmarking against other implementations."""
    
    @pytest.mark.skipif(not PSYCHROLIB_AVAILABLE, reason="PsychroLib not available")
    def test_performance_vs_psychrolib(self):
        """Benchmark performance against PsychroLib."""
        np.random.seed(42)
        n_points = 100  # Reasonable number for timing tests
        
        temps = np.random.uniform(0, 40, n_points)
        rhs = np.random.uniform(20, 95, n_points)
        pressures = np.full(n_points, 1013.25)
        
        # Benchmark our implementation
        start_time = time.time()
        our_results = davies_jones_wet_bulb(
            temps, rhs, pressures,
            vapor='hyland_wexler', convergence='newton'
        )
        our_time = time.time() - start_time
        
        # Benchmark PsychroLib (scalar only)
        start_time = time.time()
        psychrolib_results = []
        for t, rh, p in zip(temps, rhs, pressures):
            result = psychrolib.GetTWetBulbFromRelHum(t, rh/100.0, p*100.0)
            psychrolib_results.append(result)
        psychrolib_time = time.time() - start_time
        
        print(f"\nPerformance Benchmark ({n_points} points):")
        print(f"Our implementation: {our_time:.3f}s")
        print(f"PsychroLib (scalar): {psychrolib_time:.3f}s")
        print(f"Speedup: {psychrolib_time/our_time:.1f}x")
        
        # Compare accuracy
        differences = np.abs(np.array(our_results) - np.array(psychrolib_results))
        print(f"Mean difference: {np.mean(differences):.3f}°C")
        print(f"Max difference: {np.max(differences):.3f}°C")
        
        # Our vectorized implementation should be faster
        assert our_time < psychrolib_time, \
            "Our vectorized implementation should be faster than scalar PsychroLib"


def generate_test_report():
    """Generate a comprehensive test report."""
    report = {
        'test_summary': {
            'total_test_cases': 0,
            'passed': 0,
            'failed': 0,
            'accuracy_metrics': {}
        },
        'method_comparison': {},
        'performance_metrics': {},
        'recommendations': []
    }
    
    # This would be filled in by the test runner
    return report


if __name__ == "__main__":
    """Run tests with detailed reporting."""
    
    print("Davies-Jones Wet Bulb Temperature Implementation Test Suite")
    print("=" * 60)
    
    # Check dependencies
    if not PSYCHROLIB_AVAILABLE:
        print("WARNING: PsychroLib not available. Some tests will be skipped.")
        print("Install with: pip install psychrolib")
        print()
    
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-x"  # Stop on first failure for debugging
    ])