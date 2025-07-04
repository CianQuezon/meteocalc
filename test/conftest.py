"""
Professional pytest configuration for Davies-Jones wet bulb temperature tests.

This conftest.py file provides:
- Shared fixtures for test data and setup
- Custom markers for test categorization  
- Test collection and reporting configuration
- Plugin configuration and test environment setup

Author: Climate Science Research
Date: 2025
License: MIT
"""

import warnings
from typing import Dict, Generator, List, Tuple

import numpy as np
import pytest

# Suppress specific warnings during testing
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def pytest_configure(config) -> None:
    """Configure pytest with custom markers and settings.
    
    Args:
        config: pytest configuration object
    """
    # Register custom markers to avoid warnings
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers",
        "validation: marks tests as validation against reference data"
    )
    config.addinivalue_line(
        "markers",
        "extreme: marks tests with extreme conditions"
    )
    config.addinivalue_line(
        "markers",
        "requires_psychrolib: marks tests requiring PsychroLib"
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Modify test collection to add markers based on test names.
    
    Args:
        config: pytest configuration object
        items: list of collected test items
    """
    for item in items:
        # Add slow marker to performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)
        
        # Add validation marker to reference data tests
        if any(keyword in item.name for keyword in ["reference", "ashrae", "noaa"]):
            item.add_marker(pytest.mark.validation)
        
        # Add extreme marker to extreme condition tests
        if "extreme" in item.name:
            item.add_marker(pytest.mark.extreme)
        
        # Add integration marker to cross-validation tests
        if "psychrolib" in item.name:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.requires_psychrolib)


@pytest.fixture(scope="session")
def test_tolerances() -> Dict[str, float]:
    """Provide standardized test tolerances for accuracy validation.
    
    Returns:
        Dict[str, float]: Dictionary of tolerance values in degrees Celsius.
        
    Note:
        These tolerances are based on:
        - Instrument accuracy standards (±0.1°C for precision instruments)
        - Meteorological practice (±0.3°C for operational weather)
        - Engineering applications (±0.5°C for extreme conditions)
    """
    return {
        'strict': 0.1,      # High-precision scientific applications
        'moderate': 0.3,    # Standard meteorological applications  
        'extreme': 0.5,     # Extreme conditions and edge cases
        'engineering': 0.2,  # Engineering design applications
    }


@pytest.fixture(scope="session")
def vapor_pressure_methods() -> List[str]:
    """Provide list of available vapor pressure calculation methods.
    
    Returns:
        List[str]: Available vapor pressure formulations.
        
    Note:
        Methods ordered by typical accuracy and application:
        - hyland_wexler: ASHRAE/NIST standard, highest accuracy
        - goff_gratch: WMO standard, meteorological applications
        - buck: Simplified, good for operational use
        - bolton: Approximation, fast computation
    """
    return ['hyland_wexler', 'goff_gratch', 'buck', 'bolton']


@pytest.fixture(scope="session") 
def convergence_methods() -> List[str]:
    """Provide list of available numerical convergence methods.
    
    Returns:
        List[str]: Available root-finding algorithms.
        
    Note:
        Methods ordered by typical robustness:
        - hybrid: Adaptive method selection
        - brent: Robust bracketing method
        - newton: Fast for well-behaved functions
        - halley: Higher-order Newton variant
    """
    return ['hybrid', 'brent', 'newton', 'halley']


@pytest.fixture(scope="module")
def standard_test_conditions() -> List[Tuple[float, float, float]]:
    """Provide standard atmospheric test conditions.
    
    Returns:
        List[Tuple[float, float, float]]: Test conditions as (temp_c, rh_percent, pressure_hpa).
        
    Note:
        Conditions selected to represent:
        - Typical meteorological conditions
        - HVAC design conditions  
        - Laboratory standard conditions
        - Boundary conditions for validation
    """
    return [
        (20.0, 50.0, 1013.25),   # Standard laboratory conditions
        (25.0, 60.0, 1013.25),   # Typical comfort conditions
        (0.0, 100.0, 1013.25),   # Freezing point, saturated
        (35.0, 80.0, 1013.25),   # Hot, humid conditions
        (10.0, 30.0, 850.0),     # High altitude, dry
        (-10.0, 70.0, 1013.25),  # Cold, moderate humidity
    ]


@pytest.fixture(scope="function")
def random_test_data() -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Generate random test data with reproducible seed.
    
    Yields:
        Tuple of numpy arrays: (temperatures, relative_humidities, pressures)
        
    Note:
        Uses fixed seed for reproducible testing. Data ranges chosen
        to represent realistic atmospheric conditions.
    """
    # Set reproducible seed
    np.random.seed(42)
    
    # Generate realistic atmospheric data
    n_points = 100
    temperatures = np.random.uniform(-30, 50, n_points)  # °C
    relative_humidities = np.random.uniform(5, 100, n_points)  # %
    pressures = np.random.uniform(700, 1050, n_points)  # hPa
    
    yield temperatures, relative_humidities, pressures
    
    # Cleanup if needed
    pass


@pytest.fixture(scope="class")
def performance_test_data() -> Dict[str, np.ndarray]:
    """Generate larger dataset for performance testing.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary containing test arrays.
        
    Note:
        Larger dataset specifically for benchmarking vectorized
        operations and performance validation.
    """
    np.random.seed(123)  # Different seed for performance tests
    
    n_points = 10000
    return {
        'temperatures': np.random.uniform(-20, 45, n_points),
        'humidities': np.random.uniform(10, 95, n_points), 
        'pressures': np.random.uniform(800, 1050, n_points),
        'size': n_points
    }


@pytest.fixture(scope="function")
def capture_warnings():
    """Capture and validate warnings during test execution.
    
    Yields:
        warnings.catch_warnings context manager for warning inspection.
        
    Note:
        Useful for testing that appropriate warnings are issued
        for edge cases or deprecated functionality.
    """
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Perform global test environment setup and teardown.
    
    Yields:
        None: Performs setup/teardown without returning data.
        
    Note:
        Autouse fixture that runs once per test session to configure
        the testing environment consistently.
    """
    # Setup phase
    print("\nInitializing Davies-Jones wet bulb test environment...")
    
    # Configure numpy for consistent behavior
    np.seterr(all='raise', under='ignore', invalid='ignore')
    
    # Set pandas display options for test output
    try:
        import pandas as pd
        pd.set_option('display.precision', 3)
        pd.set_option('display.width', 100)
    except ImportError:
        pass
    
    # Validate test dependencies
    required_modules = ['numpy', 'pytest']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        pytest.fail(f"Required modules missing: {missing_modules}")
    
    yield  # Tests run here
    
    # Teardown phase
    print("\nTest environment cleanup completed.")


# Custom assertion helpers
def assert_within_tolerance(actual: float, expected: float, tolerance: float, 
                          context: str = "") -> None:
    """Custom assertion for tolerance-based comparisons.
    
    Args:
        actual: Calculated value
        expected: Expected reference value  
        tolerance: Acceptable difference in same units
        context: Additional context for failure messages
        
    Raises:
        AssertionError: If difference exceeds tolerance
    """
    difference = abs(actual - expected)
    assert difference <= tolerance, \
        f"Value {actual} not within {tolerance} of expected {expected} " \
        f"(difference: {difference:.4f}). Context: {context}"


def assert_physical_constraints(wet_bulb: float, dry_bulb: float, 
                              humidity: float) -> None:
    """Assert that wet bulb temperature satisfies physical constraints.
    
    Args:
        wet_bulb: Calculated wet bulb temperature
        dry_bulb: Dry bulb temperature
        humidity: Relative humidity in percent
        
    Raises:
        AssertionError: If physical constraints are violated
    """
    # Wet bulb should not exceed dry bulb (except at 100% RH due to rounding)
    if humidity < 99.9:
        assert wet_bulb <= dry_bulb, \
            f"Wet bulb ({wet_bulb:.2f}°C) exceeds dry bulb ({dry_bulb:.2f}°C) " \
            f"at {humidity:.1f}% RH"
    
    # Wet bulb should be reasonable relative to dry bulb
    assert wet_bulb > dry_bulb - 30, \
        f"Wet bulb ({wet_bulb:.2f}°C) unreasonably low compared to " \
        f"dry bulb ({dry_bulb:.2f}°C)"
    
    # At very high humidity, wet bulb should approach dry bulb
    if humidity > 98:
        assert abs(wet_bulb - dry_bulb) < 1.0, \
            f"At high humidity ({humidity:.1f}%), wet bulb should be close to dry bulb"


# Parametrize helpers for common test patterns
def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on test function names.
    
    Args:
        metafunc: pytest metafunction object for test parametrization
        
    Note:
        This function automatically parametrizes tests based on naming
        conventions, reducing boilerplate code in test definitions.
    """
    # Parametrize method combination tests
    if "vapor_method" in metafunc.fixturenames and "convergence_method" in metafunc.fixturenames:
        vapor_methods = ['bolton', 'goff_gratch', 'buck', 'hyland_wexler']
        convergence_methods = ['newton', 'brent', 'halley', 'hybrid']
        
        # Create all combinations for comprehensive testing
        combinations = [(v, c) for v in vapor_methods for c in convergence_methods]
        metafunc.parametrize(
            "vapor_method,convergence_method", 
            combinations,
            ids=[f"{v}+{c}" for v, c in combinations]
        )
    
    # Parametrize tolerance tests
    if "tolerance_level" in metafunc.fixturenames:
        tolerance_levels = ['strict', 'moderate', 'extreme']
        metafunc.parametrize("tolerance_level", tolerance_levels)


# Test data validation helpers
def validate_test_data_integrity(data_frame) -> bool:
    """Validate integrity of reference test data.
    
    Args:
        data_frame: pandas DataFrame containing test data
        
    Returns:
        bool: True if data passes validation checks
        
    Raises:
        ValueError: If data fails validation
    """
    required_columns = ['temp_c', 'rh_percent', 'pressure_hpa', 'expected_wb_c']
    
    # Check required columns exist
    missing_columns = [col for col in required_columns if col not in data_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for valid data ranges
    if (data_frame['temp_c'] < -60).any() or (data_frame['temp_c'] > 70).any():
        raise ValueError("Temperature values outside reasonable range (-60 to 70°C)")
    
    if (data_frame['rh_percent'] < 0).any() or (data_frame['rh_percent'] > 100).any():
        raise ValueError("Relative humidity values outside valid range (0-100%)")
    
    if (data_frame['pressure_hpa'] < 300).any() or (data_frame['pressure_hpa'] > 1100).any():
        raise ValueError("Pressure values outside reasonable range (300-1100 hPa)")
    
    # Check for NaN values
    if data_frame[required_columns].isnull().any().any():
        raise ValueError("Test data contains NaN values")
    
    return True


# Performance measurement utilities
@pytest.fixture
def performance_timer():
    """Provide high-resolution timing for performance tests.
    
    Yields:
        callable: Timer function that returns elapsed time in seconds
    """
    import time
    
    def timer():
        return time.perf_counter()
    
    yield timer
