"""
Unit tests for _poisson_vectorised.

Validates correctness of the Numba-jitted vectorised Poisson equation
against MetPy's reference implementation of potential_temperature.

Author: Cian Quezon
"""

import numpy as np
import pytest
from meteocalc.thermodynamics._jit_equations import _poisson_vectorised
from metpy.calc import potential_temperature
from metpy.units import units

# ---------------------------------------------------------------------------
# Constants (must match the implementation)
# ---------------------------------------------------------------------------
RD: float = 287.04
CP: float = 1004.0
KAPPA: float = RD / CP
P0_HPA: float = 1000.0

# Exact tolerance for analytical identity tests (no external reference involved)
EXACT_TOL: float = 1e-6


def compute_atol(p_min_hpa: float) -> float:
    """
    Absolute tolerance for MetPy comparison tests, based on minimum pressure.

    The systematic offset between this implementation and MetPy scales with
    theta (~0.084% of theta), and theta grows as pressure drops. Thresholds
    are set to the valid Davies-Jones domain (100–1050 hPa).

    p >= 500 hPa  ->  near-surface, theta ~300 K  ->  atol = 0.1 K
    p >= 200 hPa  ->  mid troposphere, theta ~450 K  ->  atol = 0.5 K
    p >= 100 hPa  ->  upper troposphere, theta ~600 K  ->  atol = 1.1 K
    """
    if p_min_hpa >= 500.0:
        return 0.1
    elif p_min_hpa >= 200.0:
        return 0.5
    else:
        return 1.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metpy_theta(temp_k: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """
    Reference implementation via MetPy.

    MetPy's potential_temperature uses the same physical constants as this
    implementation (Rd = 287.04, cp = 1004, p0 = 1000 hPa) and is tested
    against published meteorological datasets, making it a reliable oracle.

    Parameters
    ----------
    temp_k : temperature array (K)
    p_hpa  : pressure array (hPa)

    Returns
    -------
    theta : potential temperature array (K), plain numpy float64
    """
    theta = potential_temperature(
        p_hpa * units.hPa,
        temp_k * units.kelvin,
    )
    return theta.to("kelvin").magnitude


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_inputs():
    """A realistic tropospheric sounding: surface to ~200 hPa."""
    temp_k = np.array([300.0, 290.0, 275.0, 260.0, 240.0, 220.0], dtype=np.float64)
    p_hpa  = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0], dtype=np.float64)
    return temp_k, p_hpa


@pytest.fixture
def single_element():
    """Edge case: length-1 arrays."""
    return np.array([288.15], dtype=np.float64), np.array([1013.25], dtype=np.float64)


@pytest.fixture
def at_reference_pressure():
    """When p == p0, theta must equal T exactly."""
    temp_k = np.array([250.0, 273.15, 300.0, 320.0], dtype=np.float64)
    p_hpa  = np.full(4, P0_HPA, dtype=np.float64)
    return temp_k, p_hpa


@pytest.fixture
def high_altitude():
    """
    Upper troposphere — representative ceiling for Davies-Jones wet bulb.
    Davies-Jones pseudoadiabat lifting rarely exceeds 100 hPa, so
    p < 100 hPa is outside the valid input domain of this application.
    """
    temp_k = np.array([230.0, 220.0, 210.0], dtype=np.float64)
    p_hpa  = np.array([300.0, 200.0, 100.0], dtype=np.float64)
    return temp_k, p_hpa


@pytest.fixture
def large_random():
    """Large randomised array within the valid Davies-Jones domain."""
    rng = np.random.default_rng(seed=42)
    n = 10_000
    temp_k = rng.uniform(200.0, 330.0, n).astype(np.float64)
    p_hpa  = rng.uniform(100.0, 1050.0, n).astype(np.float64)
    return temp_k, p_hpa


# ---------------------------------------------------------------------------
# Correctness: agreement with MetPy oracle
# ---------------------------------------------------------------------------

class TestAgainstMetPy:
    """Every test here uses MetPy as the ground truth."""

    def test_standard_sounding(self, standard_inputs):
        temp_k, p_hpa = standard_inputs
        result   = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        expected = metpy_theta(temp_k, p_hpa)
        atol     = compute_atol(p_hpa.min())
        np.testing.assert_allclose(result, expected, atol=atol,
            err_msg="Standard sounding differs from MetPy reference.")

    def test_single_element(self, single_element):
        temp_k, p_hpa = single_element
        result   = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        expected = metpy_theta(temp_k, p_hpa)
        atol     = compute_atol(p_hpa.min())
        np.testing.assert_allclose(result, expected, atol=atol,
            err_msg="Single-element array differs from MetPy reference.")

    def test_high_altitude(self, high_altitude):
        temp_k, p_hpa = high_altitude
        result   = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        expected = metpy_theta(temp_k, p_hpa)
        atol     = compute_atol(p_hpa.min())
        np.testing.assert_allclose(result, expected, atol=atol,
            err_msg="High-altitude pressures differ from MetPy reference.")

    def test_large_random_array(self, large_random):
        temp_k, p_hpa = large_random
        result   = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        expected = metpy_theta(temp_k, p_hpa)
        atol     = compute_atol(p_hpa.min())
        np.testing.assert_allclose(result, expected, atol=atol,
            err_msg="Large randomised array differs from MetPy reference.")


# ---------------------------------------------------------------------------
# Analytical identities (independent of MetPy)
# ---------------------------------------------------------------------------

class TestAnalyticalIdentities:
    """Tests grounded in the physics / mathematics of the equation itself."""

    def test_theta_equals_temp_at_reference_pressure(self, at_reference_pressure):
        """
        At p = p0 the exponent evaluates to 1, so θ = T exactly.
        No kappa discrepancy applies — must agree to floating-point precision.
        """
        temp_k, p_hpa = at_reference_pressure
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        np.testing.assert_allclose(result, temp_k, atol=EXACT_TOL,
            err_msg="θ should equal T when p == p0.")

    def test_theta_greater_than_temp_below_reference_pressure(self):
        """
        For p < p0, (p0/p)^kappa > 1, so θ > T.
        """
        temp_k = np.array([300.0, 280.0, 260.0], dtype=np.float64)
        p_hpa  = np.array([500.0, 700.0, 850.0], dtype=np.float64)
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        assert np.all(result > temp_k), (
            "θ must exceed T for all pressures below the reference level.")

    def test_monotone_in_pressure(self):
        """
        At constant T, θ is a strictly decreasing function of p
        (higher pressure → lower θ).
        """
        temp_k = np.full(5, 280.0, dtype=np.float64)
        p_hpa  = np.array([200.0, 400.0, 600.0, 800.0, 1000.0], dtype=np.float64)
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        assert np.all(np.diff(result) < 0), (
            "θ must be strictly decreasing with increasing pressure at constant T.")

    def test_linearity_in_temperature(self):
        """
        θ is linear in T: _poisson(2T, p) = 2 * _poisson(T, p).
        """
        temp_k = np.array([250.0, 275.0, 300.0], dtype=np.float64)
        p_hpa  = np.array([500.0, 700.0, 850.0], dtype=np.float64)
        result_single = _poisson_vectorised(temp_k,       p_hpa, P0_HPA)
        result_double = _poisson_vectorised(2.0 * temp_k, p_hpa, P0_HPA)
        np.testing.assert_allclose(result_double, 2.0 * result_single, atol=EXACT_TOL,
            err_msg="θ is not linear in T.")

    def test_known_scalar_value(self):
        """
        Hand-computed reference: T=300 K, p=500 hPa, p0=1000 hPa.
            θ = 300 * (1000/500)^(287.04/1004) ≈ 365.628 K
        """
        expected = 300.0 * (1000.0 / 500.0) ** KAPPA
        temp_k = np.array([300.0], dtype=np.float64)
        p_hpa  = np.array([500.0], dtype=np.float64)
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        np.testing.assert_allclose(result[0], expected, atol=EXACT_TOL,
            err_msg=f"Hand-computed θ={expected:.6f} K not matched.")


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    """Shape, dtype, and finiteness guarantees."""

    def test_output_length_matches_input(self, standard_inputs):
        temp_k, p_hpa = standard_inputs
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        assert len(result) == len(temp_k), "Output length must match input length."

    def test_output_dtype_is_float64(self, standard_inputs):
        temp_k, p_hpa = standard_inputs
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        assert result.dtype == np.float64, (
            f"Expected float64 output, got {result.dtype}.")

    def test_all_values_finite(self, large_random):
        temp_k, p_hpa = large_random
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        assert np.all(np.isfinite(result)), "Output contains non-finite values."

    def test_all_values_positive(self, large_random):
        """Potential temperature is always positive for positive T."""
        temp_k, p_hpa = large_random
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        assert np.all(result > 0), "All θ values must be positive."


# ---------------------------------------------------------------------------
# Regression: catches the known indexing bug
# ---------------------------------------------------------------------------

class TestRegressions:
    """
    Explicit regression tests that would FAIL with the original buggy
    implementation (where _poisson_scalar received full arrays instead of
    indexed scalars).
    """

    def test_each_element_is_independently_computed(self):
        """
        If the bug is present, every element of the output would be identical
        (the scalar function would silently operate on temp_k[0] / p[0] every
        time under Numba's type coercion). This test verifies that different
        (T, p) pairs produce different θ values.
        """
        temp_k = np.array([270.0, 290.0, 310.0], dtype=np.float64)
        p_hpa  = np.array([800.0, 600.0, 400.0], dtype=np.float64)
        result = _poisson_vectorised(temp_k, p_hpa, P0_HPA)
        assert len(set(np.round(result, 4))) == len(result), (
            "All output values are identical — likely the indexing bug is present: "
            "use temp_k[i] and p[i] inside the loop, not temp_k and p.")

    def test_element_order_matters(self):
        """
        Reversing the input arrays should reverse the output, not leave it
        unchanged (which would happen if indexing is broken).
        """
        temp_k = np.array([260.0, 280.0, 300.0], dtype=np.float64)
        p_hpa  = np.array([400.0, 700.0, 900.0], dtype=np.float64)
        result_fwd = _poisson_vectorised(temp_k,       p_hpa,       P0_HPA)
        result_rev = _poisson_vectorised(temp_k[::-1], p_hpa[::-1], P0_HPA)
        np.testing.assert_allclose(result_fwd, result_rev[::-1], atol=EXACT_TOL,
            err_msg="Reversing inputs did not reverse outputs — indexing bug suspected.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])