"""
Comprehensive unit tests for BoltonLclEquation
===============================================
Cross-validated against MetPy's LCL solver (Romps 2017) to ensure accuracy.

BoltonLclEquation implements Bolton (1980) eq. 15 for LCL temperature:

    T_LCL = 56 + 1 / [ 1/(Td - 56) + ln(T/Td) / 800 ]

and the dry adiabatic Poisson relation for LCL pressure:

    p_LCL = p_surface * (T_LCL / T_surface) ^ (cpd / Rd)

Tolerances
----------
- vs MetPy (Romps 2017 exact):
    Near-saturated (ΔT < 3 K):    < 0.3 K
    Typical (ΔT 3–10 K):          < 0.8 K
    Dry (ΔT 10–20 K):             < 1.5 K
    Very dry (ΔT > 20 K):         < 2.5 K
- Convergence:                    100% for physically valid inputs
- Scalar/array consistency:       < 1e-10 K

Dependencies
------------
    pip install pytest numpy numba metpy
"""

import math

import numpy as np
import pytest

from meteocalc.lcl._enums import LclEquationName, CalculationMethod
from meteocalc.lcl._lcl_equation import BoltonLclEquation
from meteocalc.shared.constants import Rd, cpd

# ---------------------------------------------------------------------------
# ---- MetPy helper ---------------------------------------------------------
# ---------------------------------------------------------------------------

metpy = pytest.importorskip("metpy", reason="MetPy not installed — skipping MetPy tests")


def _metpy_lcl(pressure_hpa: float, temp_k: float, dewpoint_k: float):
    """Return MetPy LCL (temp_K, pressure_hPa) using Romps (2017)."""
    from metpy.calc import lcl
    from metpy.units import units

    lcl_p, lcl_t = lcl(
        pressure_hpa * units.hPa,
        temp_k       * units.kelvin,
        dewpoint_k   * units.kelvin,
    )
    return (
        float(lcl_t.to("kelvin").magnitude),
        float(lcl_p.to("hPa").magnitude),
    )


# ---------------------------------------------------------------------------
# ---- Fixtures -------------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eq() -> BoltonLclEquation:
    """Shared BoltonLclEquation instance."""
    return BoltonLclEquation()


# ---------------------------------------------------------------------------
# ---- Test data ------------------------------------------------------------
# ---------------------------------------------------------------------------

# (label, pressure_hpa, temp_k, dewpoint_k)
STANDARD_CASES = [
    ("standard_midlat",    1013.25, 293.15, 285.15),  # 20°C / 12°C — 8 K depression
    ("warm_humid",         1000.0,  303.15, 298.15),  # 30°C / 25°C — 5 K depression
    ("cool_dry",           1013.25, 278.15, 268.15),  # 5°C / -5°C  — 10 K depression
    ("near_saturated",     1013.25, 290.15, 289.15),  # 1 K depression
    ("high_elevation",     850.0,   285.15, 278.15),  # 12°C / 5°C
    ("summer_continental", 1000.0,  300.15, 290.15),  # 27°C / 17°C — 10 K depression
    ("autumn_uk",          1013.25, 283.15, 278.15),  # 10°C / 5°C  — 5 K depression
    ("moderate_dry",       1013.25, 295.15, 282.15),  # 22°C / 9°C  — 13 K depression
]

# Bolton valid range: 233.15–323.15 K, depression < 30 K
DEPRESSION_SWEEP = [
    (273.15 + 15, [1, 2, 5, 10, 15, 20]),  # 15°C base
    (273.15 + 25, [1, 2, 5, 10, 15, 20]),  # 25°C base
]

# Tiered MetPy tolerances based on Bolton (1980) stated accuracy
def _metpy_tol(depression: float) -> float:
    if depression < 3.0:
        return 0.3
    elif depression < 10.0:
        return 0.8
    elif depression < 20.0:
        return 1.5
    return 2.5


# ===========================================================================
# Section 1 — Class attributes
# ===========================================================================

class TestClassAttributes:
    """BoltonLclEquation must expose correct class-level metadata."""

    def test_name_is_bolton(self, eq):
        """name must be LclEquationName.BOLTON."""
        assert eq.name == LclEquationName.BOLTON

    def test_calculation_method_is_approximation(self, eq):
        """calculation_method must be CalculationMethod.APPROXIMATION."""
        assert eq.calculation_method == CalculationMethod.APPROXIMATION

    def test_is_instance_of_base_class(self, eq):
        """Must be an instance of LiftingCondensationLevelEquation."""
        from meteocalc.lcl._lcl_equation import LiftingCondensationLevelEquation
        assert isinstance(eq, LiftingCondensationLevelEquation)

    def test_name_value(self, eq):
        """LclEquationName.BOLTON must have value 'bolton'."""
        assert eq.name.value == "bolton"

    def test_calculation_method_value(self, eq):
        """CalculationMethod.APPROXIMATION must have value 'approximation'."""
        assert eq.calculation_method.value == "approximation"


# ===========================================================================
# Section 2 — Physical constraints
# ===========================================================================

class TestPhysicalConstraints:
    """LCL results must satisfy fundamental atmospheric physics."""

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_below_surface_temp(self, eq, label, pressure, temp_k, dewpoint_k):
        """T_LCL ≤ T_surface — parcel cools on dry adiabatic ascent."""
        t_lcl, _ = eq.calculate(temp_k, dewpoint_k, pressure)
        assert t_lcl <= temp_k + 1e-9, (
            f"[{label}] T_LCL={t_lcl:.4f} K > T_surface={temp_k:.4f} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_pressure_below_surface_pressure(self, eq, label, pressure, temp_k, dewpoint_k):
        """p_LCL ≤ p_surface — LCL is above the surface."""
        _, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure)
        assert p_lcl <= pressure + 1e-9, (
            f"[{label}] p_LCL={p_lcl:.4f} hPa > p_surface={pressure:.4f} hPa"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_temp_above_absolute_zero(self, eq, label, pressure, temp_k, dewpoint_k):
        """T_LCL must be physically positive."""
        t_lcl, _ = eq.calculate(temp_k, dewpoint_k, pressure)
        assert t_lcl > 0.0, f"[{label}] T_LCL={t_lcl:.4f} K ≤ 0 K"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_lcl_pressure_above_zero(self, eq, label, pressure, temp_k, dewpoint_k):
        """p_LCL must be positive."""
        _, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure)
        assert p_lcl > 0.0, f"[{label}] p_LCL={p_lcl:.4f} hPa ≤ 0"

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_results_are_finite(self, eq, label, pressure, temp_k, dewpoint_k):
        """Both T_LCL and p_LCL must be finite — no NaN or inf."""
        t_lcl, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure)
        assert math.isfinite(t_lcl),  f"[{label}] T_LCL is not finite: {t_lcl}"
        assert math.isfinite(p_lcl),  f"[{label}] p_LCL is not finite: {p_lcl}"

    def test_poisson_relation_correct(self, eq):
        """
        p_LCL must satisfy the Poisson relation exactly:
            p_LCL = p_surface * (T_LCL / T_surface) ^ (cpd / Rd)
        Validates that the pressure calculation is internally consistent
        with the returned temperature.
        """
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25
        t_lcl, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure_hpa)

        expected_p = pressure_hpa * (t_lcl / temp_k) ** (cpd / Rd)
        assert abs(p_lcl - expected_p) < 1e-6, (
            f"p_LCL={p_lcl:.6f} hPa does not satisfy Poisson relation "
            f"(expected {expected_p:.6f} hPa, Δ={abs(p_lcl - expected_p):.2e})"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_poisson_relation_all_cases(self, eq, label, pressure, temp_k, dewpoint_k):
        """Poisson relation must hold for all standard cases."""
        t_lcl, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure)
        expected_p   = pressure * (t_lcl / temp_k) ** (cpd / Rd)
        assert abs(p_lcl - expected_p) < 1e-6, (
            f"[{label}] Poisson relation violated: "
            f"p_LCL={p_lcl:.6f} vs expected {expected_p:.6f} hPa"
        )


# ===========================================================================
# Section 3 — API behaviour (scalar / array)
# ===========================================================================

class TestAPIBehaviour:
    """Scalar inputs return scalars; array inputs return arrays."""

    def test_scalar_input_returns_floats(self, eq):
        """Scalar inputs must return Python float, not numpy scalar."""
        t_lcl, p_lcl = eq.calculate(293.15, 285.15, 1013.25)
        assert isinstance(t_lcl, float), f"Expected float, got {type(t_lcl)}"
        assert isinstance(p_lcl, float), f"Expected float, got {type(p_lcl)}"

    def test_array_input_returns_ndarrays(self, eq):
        """Array inputs must return numpy ndarrays."""
        t_lcl, p_lcl = eq.calculate(
            np.array([293.15, 303.15]),
            np.array([285.15, 298.15]),
            np.array([1013.25, 1000.0]),
        )
        assert isinstance(t_lcl, np.ndarray)
        assert isinstance(p_lcl, np.ndarray)

    def test_array_output_dtype_float64(self, eq):
        """Array outputs must be float64."""
        t_lcl, p_lcl = eq.calculate(
            np.array([293.15, 303.15]),
            np.array([285.15, 298.15]),
            np.full(2, 1013.25),
        )
        assert t_lcl.dtype == np.float64
        assert p_lcl.dtype == np.float64

    def test_array_output_shape_matches_input(self, eq):
        """Output shape must match input array shape."""
        n = 5
        t_lcl, p_lcl = eq.calculate(
            np.full(n, 293.15),
            np.full(n, 285.15),
            np.full(n, 1013.25),
        )
        assert t_lcl.shape == (n,)
        assert p_lcl.shape == (n,)

    def test_scalar_array_consistency(self, eq):
        """Scalar and array calls with the same inputs must give identical results."""
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25

        t_scalar, p_scalar = eq.calculate(temp_k, dewpoint_k, pressure_hpa)
        t_array,  p_array  = eq.calculate(
            np.array([temp_k]),
            np.array([dewpoint_k]),
            np.array([pressure_hpa]),
        )

        assert abs(t_scalar - float(t_array[0])) < 1e-10, (
            f"Scalar={t_scalar:.10f} K vs array={float(t_array[0]):.10f} K"
        )
        assert abs(p_scalar - float(p_array[0])) < 1e-10

    def test_returns_two_values(self, eq):
        """Must return exactly two values for both scalar and array input."""
        scalar_result = eq.calculate(293.15, 285.15, 1013.25)
        array_result  = eq.calculate(
            np.array([293.15]), np.array([285.15]), np.array([1013.25])
        )
        assert len(scalar_result) == 2
        assert len(array_result)  == 2

    def test_mixed_scalar_array_broadcast(self, eq):
        """Mixed scalar/array inputs must broadcast correctly."""
        n = 4
        t_lcl, p_lcl = eq.calculate(
            np.full(n, 293.15),   # array
            285.15,               # scalar — broadcast to (4,)
            1013.25,              # scalar — broadcast to (4,)
        )
        assert t_lcl.shape == (n,)
        assert p_lcl.shape == (n,)

    def test_2d_array_shape_preserved(self, eq):
        """2-D array inputs must return 2-D arrays of the same shape."""
        shape = (3, 4)
        t_lcl, p_lcl = eq.calculate(
            np.full(shape, 293.15),
            np.full(shape, 285.15),
            np.full(shape, 1013.25),
        )
        assert t_lcl.shape == shape
        assert p_lcl.shape == shape

    def test_deterministic_repeated_calls(self, eq):
        """Results must be identical on repeated calls — JIT is deterministic."""
        results = [eq.calculate(293.15, 285.15, 1013.25) for _ in range(5)]
        for t, p in results:
            assert t == results[0][0]
            assert p == results[0][1]


# ===========================================================================
# Section 4 — Monotonicity
# ===========================================================================

class TestMonotonicity:
    """LCL results must respond correctly to changes in inputs."""

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_temp(self, eq, base_temp, depressions):
        """
        Increasing dewpoint depression must lower T_LCL — a drier parcel
        must ascend further before reaching saturation.
        """
        t_prev = None
        for dep in sorted(depressions):
            t_lcl, _ = eq.calculate(base_temp, base_temp - dep, 1013.25)
            if t_prev is not None:
                assert t_lcl < t_prev, (
                    f"T={base_temp:.2f} K, dep={dep} K: T_LCL did not decrease "
                    f"(got {t_lcl:.4f} K, prev={t_prev:.4f} K)"
                )
            t_prev = t_lcl

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_larger_depression_gives_lower_lcl_pressure(self, eq, base_temp, depressions):
        """Increasing depression must lower p_LCL — LCL is higher in atmosphere."""
        p_prev = None
        for dep in sorted(depressions):
            _, p_lcl = eq.calculate(base_temp, base_temp - dep, 1013.25)
            if p_prev is not None:
                assert p_lcl < p_prev, (
                    f"T={base_temp:.2f} K, dep={dep} K: p_LCL did not decrease "
                    f"(got {p_lcl:.4f} hPa, prev={p_prev:.4f} hPa)"
                )
            p_prev = p_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_warmer_parcel_gives_warmer_lcl_temp(self, eq, depression):
        """At fixed depression, warmer parcel must give warmer T_LCL."""
        temps  = [278.15, 283.15, 288.15, 293.15, 298.15]
        t_prev = None
        for t in temps:
            t_lcl, _ = eq.calculate(t, t - depression, 1013.25)
            if t_prev is not None:
                assert t_lcl > t_prev, (
                    f"depression={depression} K: warmer parcel did not give "
                    f"warmer T_LCL (got {t_lcl:.4f} K, prev={t_prev:.4f} K)"
                )
            t_prev = t_lcl

    @pytest.mark.parametrize("depression", [2, 5, 10])
    def test_higher_surface_pressure_raises_lcl_pressure(self, eq, depression):
        """Higher surface pressure must give higher p_LCL at fixed T and Td."""
        temp_k     = 293.15
        dewpoint_k = temp_k - depression
        pressures  = [850.0, 900.0, 950.0, 1013.25]
        p_prev     = None
        for p in pressures:
            _, p_lcl = eq.calculate(temp_k, dewpoint_k, p)
            if p_prev is not None:
                assert p_lcl > p_prev - 1e-6, (
                    f"depression={depression} K, P={p} hPa: "
                    f"higher surface pressure did not give higher p_LCL"
                )
            p_prev = p_lcl


# ===========================================================================
# Section 5 — Cross-validation against MetPy (Romps 2017)
# ===========================================================================

class TestMetPyCrossValidation:
    """
    Validate Bolton LCL against MetPy's Romps (2017) exact solver.

    MetPy's LCL uses an exact analytical solution (Romps 2017) with
    error ~1e-10 K, so all discrepancy here is Bolton's approximation
    error. Tolerances are tiered by dewpoint depression following
    Bolton (1980) Table 1.
    """

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_temp_agrees_with_metpy(self, eq, label, pressure, temp_k, dewpoint_k):
        """T_LCL must agree with MetPy within tiered Bolton tolerances."""
        depression = temp_k - dewpoint_k
        tol        = _metpy_tol(depression)

        t_bolton, _    = eq.calculate(temp_k, dewpoint_k, pressure)
        t_metpy, _     = _metpy_lcl(pressure, temp_k, dewpoint_k)
        diff           = abs(t_bolton - t_metpy)

        assert diff < tol, (
            f"[{label}] Bolton={t_bolton:.4f} K  MetPy={t_metpy:.4f} K  "
            f"Δ={diff:.4f} K  depression={depression:.1f} K  tol={tol} K"
        )

    @pytest.mark.parametrize("label,pressure,temp_k,dewpoint_k", STANDARD_CASES)
    def test_pressure_agrees_with_metpy(self, eq, label, pressure, temp_k, dewpoint_k):
        """p_LCL must agree with MetPy within a pressure-equivalent tolerance."""
        depression = temp_k - dewpoint_k
        tol_k      = _metpy_tol(depression)

        _, p_bolton    = eq.calculate(temp_k, dewpoint_k, pressure)
        _, p_metpy     = _metpy_lcl(pressure, temp_k, dewpoint_k)

        # Convert temperature tolerance to approximate pressure tolerance
        # dp/dT ≈ p_LCL * (cpd/Rd) / T_LCL ≈ 3.5 hPa/K at typical conditions
        tol_hpa = tol_k * 4.0
        diff    = abs(p_bolton - p_metpy)

        assert diff < tol_hpa, (
            f"[{label}] Bolton={p_bolton:.4f} hPa  MetPy={p_metpy:.4f} hPa  "
            f"Δ={diff:.4f} hPa  tol={tol_hpa:.2f} hPa"
        )

    @pytest.mark.parametrize("base_temp,depressions", DEPRESSION_SWEEP)
    def test_depression_sweep_agrees_with_metpy(self, eq, base_temp, depressions):
        """Sweep dewpoint depressions and compare T_LCL against MetPy."""
        for dep in depressions:
            td  = base_temp - dep
            tol = _metpy_tol(dep)

            t_bolton, _ = eq.calculate(base_temp, td, 1013.25)
            t_metpy, _  = _metpy_lcl(1013.25, base_temp, td)
            diff        = abs(t_bolton - t_metpy)

            assert diff < tol, (
                f"T={base_temp:.2f} K, dep={dep} K: "
                f"Bolton={t_bolton:.4f} K  MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )

    def test_near_saturated_tight_agreement(self, eq):
        """Near-saturated (ΔT < 2 K) — Bolton is most accurate here."""
        cases = [
            (1013.25, 293.15, 292.15),   # 1 K depression
            (1013.25, 293.15, 291.65),   # 1.5 K depression
            (1013.25, 303.15, 302.15),   # 1 K, warm
        ]
        for p, t, td in cases:
            t_bolton, _ = eq.calculate(t, td, p)
            t_metpy, _  = _metpy_lcl(p, t, td)
            diff        = abs(t_bolton - t_metpy)
            assert diff < 0.3, (
                f"Near-saturated: Bolton={t_bolton:.4f} K  MetPy={t_metpy:.4f} K  "
                f"Δ={diff:.4f} K (tol=0.3 K)"
            )

    def test_tropical_conditions_agree(self, eq):
        """Tropical warm/humid conditions — Bolton tested at extremes."""
        cases = [
            (1010.0, 303.15, 298.15),   # 30°C / 25°C
            (1010.0, 308.15, 303.15),   # 35°C / 30°C
        ]
        for p, t, td in cases:
            dep         = t - td
            tol         = _metpy_tol(dep)
            t_bolton, _ = eq.calculate(t, td, p)
            t_metpy, _  = _metpy_lcl(p, t, td)
            diff        = abs(t_bolton - t_metpy)
            assert diff < tol, (
                f"Tropical: Bolton={t_bolton:.4f} K  MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )

    def test_large_array_agrees_with_metpy_elementwise(self, eq):
        """
        Element-wise comparison of array output against MetPy for
        all standard cases simultaneously.
        """
        temp_arr     = np.array([c[2] for c in STANDARD_CASES])
        dewpoint_arr = np.array([c[3] for c in STANDARD_CASES])
        pressure_arr = np.array([c[1] for c in STANDARD_CASES])

        t_bolton_arr, _ = eq.calculate(temp_arr, dewpoint_arr, pressure_arr)

        for i, (label, pressure, temp_k, dewpoint_k) in enumerate(STANDARD_CASES):
            depression  = temp_k - dewpoint_k
            tol         = _metpy_tol(depression)
            t_metpy, _  = _metpy_lcl(pressure, temp_k, dewpoint_k)
            diff        = abs(float(t_bolton_arr[i]) - t_metpy)
            assert diff < tol, (
                f"[{label}] Array element {i}: Bolton={t_bolton_arr[i]:.4f} K  "
                f"MetPy={t_metpy:.4f} K  Δ={diff:.4f} K"
            )


# ===========================================================================
# Section 6 — Numerical precision and regression
# ===========================================================================

class TestNumericalPrecision:
    """
    Hard-coded regression values seeded from actual BoltonLclEquation output.
    Any implementation change that alters results will immediately fail here.
    """

    CASES = [
        # (temp_k, dewpoint_k, pressure_hpa, expected_t_lcl, expected_p_lcl, tol_t, tol_p)
    (293.15, 285.15, 1013.25,  283.3482,  899.5646,  1e-3,  1e-2),
    (303.15, 298.15, 1000.0,   296.9371,  930.1001,  1e-3,  1e-2),
    (278.15, 268.15, 1013.25,  266.1099,  867.8913,  1e-3,  1e-2),
    (300.15, 290.15, 1000.0,   287.8506,  863.7975,  1e-3,  1e-2),
    (290.15, 289.15, 1013.25,  288.9156,  998.2457,  1e-3,  1e-2),
    ]

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_temp(
        self, eq, temp_k, dewpoint_k, pressure_hpa, exp_t, exp_p, tol_t, tol_p
    ):
        """T_LCL must match seeded regression value within tolerance."""
        t_lcl, _ = eq.calculate(temp_k, dewpoint_k, pressure_hpa)
        assert abs(t_lcl - exp_t) < tol_t, (
            f"T={temp_k}, Td={dewpoint_k}: got {t_lcl:.6f} K, "
            f"expected {exp_t:.6f} K (Δ={abs(t_lcl-exp_t):.6f} K)"
        )

    @pytest.mark.parametrize(
        "temp_k,dewpoint_k,pressure_hpa,exp_t,exp_p,tol_t,tol_p", CASES
    )
    def test_regression_pressure(
        self, eq, temp_k, dewpoint_k, pressure_hpa, exp_t, exp_p, tol_t, tol_p
    ):
        """p_LCL must match seeded regression value within tolerance."""
        _, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure_hpa)
        assert abs(p_lcl - exp_p) < tol_p, (
            f"T={temp_k}, Td={dewpoint_k}: got {p_lcl:.4f} hPa, "
            f"expected {exp_p:.4f} hPa (Δ={abs(p_lcl-exp_p):.4f} hPa)"
        )

    def test_sensitivity_to_small_dewpoint_change(self, eq):
        """
        Small perturbation in Td must produce small proportional change
        in T_LCL. Guards against numerical instability.
        """
        delta = 0.01  # 0.01 K perturbation
        t     = 293.15
        td    = 285.15

        t1, _ = eq.calculate(t, td,         1013.25)
        t2, _ = eq.calculate(t, td + delta, 1013.25)

        assert t2 > t1, "Increasing Td should increase T_LCL"
        assert abs(t2 - t1) < delta * 10, (
            f"T_LCL changed by {abs(t2-t1):.6f} K for {delta} K Td perturbation"
        )


# ===========================================================================
# Section 7 — Edge cases
# ===========================================================================

class TestEdgeCases:
    """Boundary and unusual inputs within Bolton's valid range."""

    def test_very_small_depression(self, eq):
        """0.5 K depression — nearly saturated parcel."""
        t_lcl, p_lcl = eq.calculate(293.15, 292.65, 1013.25)
        assert math.isfinite(t_lcl)
        assert math.isfinite(p_lcl)
        assert t_lcl <= 293.15

    def test_high_pressure_surface(self, eq):
        """1050 hPa surface pressure — valid but at upper bound."""
        t_lcl, p_lcl = eq.calculate(293.15, 285.15, 1050.0)
        assert math.isfinite(t_lcl)
        assert math.isfinite(p_lcl)
        assert t_lcl <= 293.15
        assert p_lcl <= 1050.0

    def test_low_pressure_surface(self, eq):
        """850 hPa — elevated terrain."""
        t_lcl, p_lcl = eq.calculate(285.15, 278.15, 850.0)
        assert math.isfinite(t_lcl)
        assert math.isfinite(p_lcl)
        assert t_lcl <= 285.15
        assert p_lcl <= 850.0

    def test_tropical_warm_humid(self, eq):
        """35°C / 30°C — hot humid tropical surface."""
        t_lcl, p_lcl = eq.calculate(308.15, 303.15, 1005.0)
        assert math.isfinite(t_lcl)
        assert math.isfinite(p_lcl)
        assert t_lcl <= 308.15

    def test_large_array_all_finite(self, eq):
        """100K random valid inputs must all produce finite results."""
        rng          = np.random.default_rng(42)
        n            = 100_000
        temp_k       = rng.uniform(243.15, 313.15, n)
        dewpoint_k   = temp_k - rng.uniform(0.5, 20.0, n)
        pressure_hpa = rng.uniform(850.0, 1013.25, n)

        t_lcl, p_lcl = eq.calculate(temp_k, dewpoint_k, pressure_hpa)

        assert np.all(np.isfinite(t_lcl)), (
            f"{np.sum(~np.isfinite(t_lcl))} non-finite T_LCL values"
        )
        assert np.all(np.isfinite(p_lcl)), (
            f"{np.sum(~np.isfinite(p_lcl))} non-finite p_LCL values"
        )
        assert np.all(t_lcl <= temp_k), "T_LCL > T_surface for some parcels"
        assert np.all(p_lcl <= pressure_hpa), "p_LCL > p_surface for some parcels"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])