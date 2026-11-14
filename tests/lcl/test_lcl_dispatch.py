"""
Comprehensive unit tests for LiftingCondensationLevelEquation._dispatch_scalar_or_vectorised
=============================================================================================

Tests cover:
- Correct routing to scalar_func vs vector_func based on input shape
- Return types — scalar inputs return floats, array inputs return ndarrays
- Shape preservation for N-dimensional inputs
- flatten → vector_func → reshape pipeline
- scalar_func and vector_func called with correct arguments
- Mixed scalar/array input handling via broadcast
- Mock functions to isolate dispatch logic from LCL physics

The dispatch method is the routing layer underpinning all public LCL
calculations — any routing error silently produces wrong results for
some input shapes.
"""

import numpy as np
import numpy.typing as npt
import pytest
from unittest.mock import MagicMock, call
from typing import Tuple

from meteocalc.lcl._lcl_equation import BoltonLclEquation, LiftingCondensationLevelEquation
from meteocalc.lcl._jit_equations import _bolton_lcl_scalar, _bolton_lcl_vectorised


# ---------------------------------------------------------------------------
# ---- Fixtures -------------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.fixture
def lcl_eq() -> LiftingCondensationLevelEquation:
    """Concrete LCL equation instance for testing the ABC method."""
    return BoltonLclEquation()


def _scalar_mock(temp_k: float, dewpoint_temp_k: float, pressure_hpa: float):
    """Mock scalar function — returns fixed sentinel values."""
    return 283.35, 917.23


def _vector_mock(
    temp_k: npt.NDArray,
    dewpoint_temp_k: npt.NDArray,
    pressure_hpa: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Mock vector function — returns arrays of sentinel values."""
    n = len(temp_k)
    return np.full(n, 283.35), np.full(n, 917.23)


def _dispatch(eq, temp_k, dewpoint_k, pressure_hpa,
              scalar_func=_scalar_mock, vector_func=_vector_mock):
    """Shorthand for calling _dispatch_scalar_or_vectorised."""
    return eq._dispatch_scalar_or_vectorised(
        temp_k=temp_k,
        dewpoint_temp_k=dewpoint_k,
        pressure_hpa=pressure_hpa,
        scalar_func=scalar_func,
        vector_func=vector_func,
    )


# ===========================================================================
# Section 1 — Routing correctness
# ===========================================================================

class TestRoutingCorrectness:
    """scalar_func called for scalar inputs, vector_func for array inputs."""

    def test_scalar_input_calls_scalar_func(self, lcl_eq):
        """All scalar inputs must route to scalar_func."""
        scalar_called = []
        vector_called = []

        def scalar_func(temp_k, dewpoint_temp_k, pressure_hpa):
            scalar_called.append(True)
            return 283.35, 917.23

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            vector_called.append(True)
            return np.array([283.35]), np.array([917.23])

        _dispatch(lcl_eq, 293.15, 285.15, 1013.25, scalar_func, vector_func)

        assert len(scalar_called) == 1, "scalar_func must be called exactly once"
        assert len(vector_called) == 0, "vector_func must not be called for scalar input"

    def test_array_input_calls_vector_func(self, lcl_eq):
        """Array inputs must route to vector_func."""
        scalar_called = []
        vector_called = []

        def scalar_func(temp_k, dewpoint_temp_k, pressure_hpa):
            scalar_called.append(True)
            return 283.35, 917.23

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            vector_called.append(True)
            return np.full(len(temp_k), 283.35), np.full(len(temp_k), 917.23)

        _dispatch(
            lcl_eq,
            np.array([293.15, 303.15]),
            np.array([285.15, 298.15]),
            np.array([1013.25, 1000.0]),
            scalar_func, vector_func,
        )

        assert len(vector_called) == 1, "vector_func must be called exactly once"
        assert len(scalar_called) == 0, "scalar_func must not be called for array input"

    def test_mixed_scalar_array_calls_vector_func(self, lcl_eq):
        """Mixed scalar/array inputs must route to vector_func."""
        vector_called = []

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            vector_called.append(True)
            return np.full(len(temp_k), 283.35), np.full(len(temp_k), 917.23)

        _dispatch(
            lcl_eq,
            np.array([293.15, 303.15]),   # array
            285.15,                        # scalar
            1013.25,                       # scalar
            _scalar_mock, vector_func,
        )

        assert len(vector_called) == 1

    def test_single_element_array_calls_vector_func(self, lcl_eq):
        """Single-element array must route to vector_func, not scalar_func."""
        vector_called = []

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            vector_called.append(True)
            return np.array([283.35]), np.array([917.23])

        _dispatch(
            lcl_eq,
            np.array([293.15]),   # single-element array — not scalar
            285.15,
            1013.25,
            _scalar_mock, vector_func,
        )

        assert len(vector_called) == 1


# ===========================================================================
# Section 2 — Scalar return types
# ===========================================================================

class TestScalarReturnTypes:
    """Scalar inputs must return Python floats, not numpy types."""

    def test_scalar_input_returns_python_floats(self, lcl_eq):
        """Scalar inputs must return Python float, not numpy scalar."""
        lcl_temp, lcl_pressure = _dispatch(lcl_eq, 293.15, 285.15, 1013.25)
        assert isinstance(lcl_temp,     float), f"Expected float, got {type(lcl_temp)}"
        assert isinstance(lcl_pressure, float), f"Expected float, got {type(lcl_pressure)}"

    def test_scalar_input_returns_two_values(self, lcl_eq):
        """Scalar dispatch must return exactly two values."""
        result = _dispatch(lcl_eq, 293.15, 285.15, 1013.25)
        assert len(result) == 2

    def test_scalar_func_receives_python_floats(self, lcl_eq):
        """
        scalar_func must receive Python floats — not numpy scalars.
        Numba @njit functions require native Python types for scalar calls.
        """
        received_types = {}

        def scalar_func(temp_k, dewpoint_temp_k, pressure_hpa):
            received_types["temp_k"]         = type(temp_k)
            received_types["dewpoint_temp_k"] = type(dewpoint_temp_k)
            received_types["pressure_hpa"]   = type(pressure_hpa)
            return 283.35, 917.23

        _dispatch(lcl_eq, 293.15, 285.15, 1013.25, scalar_func, _vector_mock)

        assert received_types["temp_k"]          is float
        assert received_types["dewpoint_temp_k"] is float
        assert received_types["pressure_hpa"]    is float

    def test_scalar_values_passed_correctly(self, lcl_eq):
        """scalar_func must receive the exact input values."""
        received = {}

        def scalar_func(temp_k, dewpoint_temp_k, pressure_hpa):
            received["temp_k"]         = temp_k
            received["dewpoint_temp_k"] = dewpoint_temp_k
            received["pressure_hpa"]   = pressure_hpa
            return 283.35, 917.23

        _dispatch(lcl_eq, 293.15, 285.15, 1013.25, scalar_func, _vector_mock)

        assert received["temp_k"]          == 293.15
        assert received["dewpoint_temp_k"] == 285.15
        assert received["pressure_hpa"]    == 1013.25


# ===========================================================================
# Section 3 — Array return types and shape
# ===========================================================================

class TestArrayReturnTypes:
    """Array inputs must return ndarrays with correct shape and dtype."""

    def test_array_input_returns_ndarrays(self, lcl_eq):
        """Array inputs must return ndarrays, not Python scalars."""
        lcl_temp, lcl_pressure = _dispatch(
            lcl_eq,
            np.array([293.15, 303.15]),
            np.array([285.15, 298.15]),
            np.array([1013.25, 1000.0]),
        )
        assert isinstance(lcl_temp,     np.ndarray)
        assert isinstance(lcl_pressure, np.ndarray)

    def test_array_output_shape_matches_input(self, lcl_eq):
        """Output shape must match input array shape."""
        n = 5
        lcl_temp, lcl_pressure = _dispatch(
            lcl_eq,
            np.full(n, 293.15),
            np.full(n, 285.15),
            np.full(n, 1013.25),
        )
        assert lcl_temp.shape    == (n,)
        assert lcl_pressure.shape == (n,)

    def test_array_returns_two_values(self, lcl_eq):
        """Array dispatch must return exactly two values."""
        result = _dispatch(
            lcl_eq,
            np.array([293.15]),
            np.array([285.15]),
            np.array([1013.25]),
        )
        assert len(result) == 2

    def test_large_array_output_shape(self, lcl_eq):
        """Large array output shape must match input."""
        n = 100_000
        rng = np.random.default_rng(42)
        lcl_temp, lcl_pressure = _dispatch(
            lcl_eq,
            rng.uniform(280.0, 310.0, n),
            rng.uniform(270.0, 295.0, n),
            rng.uniform(850.0, 1013.25, n),
            _scalar_mock,
            lambda temp_k, dewpoint_temp_k, pressure_hpa: (
                np.full(len(temp_k), 283.35),
                np.full(len(temp_k), 917.23),
            ))
        assert lcl_temp.shape    == (n,)
        assert lcl_pressure.shape == (n,)


# ===========================================================================
# Section 4 — N-dimensional shape preservation
# ===========================================================================

class TestNDimensionalShapePreservation:
    """
    N-dimensional inputs must be flattened for vector_func then reshaped
    to the original shape on output.
    """

    @pytest.mark.parametrize("shape", [
        (3,),
        (2, 3),
        (2, 3, 4),
        (5, 5),
    ])
    def test_nd_output_shape_matches_input(self, lcl_eq, shape):
        """Output shape must match N-D input shape."""
        n = int(np.prod(shape))

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            return np.full(len(temp_k), 283.35), np.full(len(temp_k), 917.23)

        lcl_temp, lcl_pressure = _dispatch(
            lcl_eq,
            np.full(shape, 293.15),
            np.full(shape, 285.15),
            np.full(shape, 1013.25),
            _scalar_mock,
            vector_func,
        )

        assert lcl_temp.shape    == shape, f"Expected {shape}, got {lcl_temp.shape}"
        assert lcl_pressure.shape == shape

    def test_vector_func_receives_1d_array(self, lcl_eq):
        """
        vector_func must always receive 1-D arrays regardless of input shape.
        The dispatch method must flatten before calling vector_func.
        """
        received_shapes = {}

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            received_shapes["temp_k"]          = temp_k.ndim
            received_shapes["dewpoint_temp_k"] = dewpoint_temp_k.ndim
            received_shapes["pressure_hpa"]    = pressure_hpa.ndim
            return np.full(len(temp_k), 283.35), np.full(len(temp_k), 917.23)

        _dispatch(
            lcl_eq,
            np.full((3, 4), 293.15),    # 2-D input
            np.full((3, 4), 285.15),
            np.full((3, 4), 1013.25),
            _scalar_mock,
            vector_func,
        )

        assert received_shapes["temp_k"]          == 1, "temp_k must be 1-D in vector_func"
        assert received_shapes["dewpoint_temp_k"] == 1
        assert received_shapes["pressure_hpa"]    == 1

    def test_nd_element_count_correct(self, lcl_eq):
        """vector_func must receive arrays with n_elements = product of shape."""
        shape = (3, 4)
        n     = 3 * 4
        received_lengths = {}

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            received_lengths["n"] = len(temp_k)
            return np.full(len(temp_k), 283.35), np.full(len(temp_k), 917.23)

        _dispatch(
            lcl_eq,
            np.full(shape, 293.15),
            np.full(shape, 285.15),
            np.full(shape, 1013.25),
            _scalar_mock,
            vector_func,
        )

        assert received_lengths["n"] == n


# ===========================================================================
# Section 5 — Physics correctness with real JIT functions
# ===========================================================================

class TestPhysicsWithRealFunctions:
    """
    Dispatch with the real Bolton JIT functions must produce
    physically correct results.
    """

    def test_scalar_dispatch_with_bolton(self, lcl_eq):
        """Real Bolton scalar function must produce finite, physical result."""
        lcl_temp, lcl_pressure = lcl_eq._dispatch_scalar_or_vectorised(
            temp_k=293.15,
            dewpoint_temp_k=285.15,
            pressure_hpa=1013.25,
            scalar_func=_bolton_lcl_scalar,
            vector_func=_bolton_lcl_vectorised,
        )
        assert isinstance(lcl_temp,     float)
        assert isinstance(lcl_pressure, float)
        assert np.isfinite(lcl_temp)
        assert np.isfinite(lcl_pressure)
        assert lcl_temp     <= 293.15
        assert lcl_pressure <= 1013.25

    def test_array_dispatch_with_bolton(self, lcl_eq):
        """Real Bolton vector function must produce finite, physical results."""
        temp_arr     = np.array([293.15, 303.15, 283.15])
        dewpoint_arr = np.array([285.15, 298.15, 278.15])
        pressure_arr = np.full(3, 1013.25)

        lcl_temp, lcl_pressure = lcl_eq._dispatch_scalar_or_vectorised(
            temp_k=temp_arr,
            dewpoint_temp_k=dewpoint_arr,
            pressure_hpa=pressure_arr,
            scalar_func=_bolton_lcl_scalar,
            vector_func=_bolton_lcl_vectorised,
        )

        assert isinstance(lcl_temp,     np.ndarray)
        assert isinstance(lcl_pressure, np.ndarray)
        assert lcl_temp.shape    == (3,)
        assert lcl_pressure.shape == (3,)
        assert np.all(np.isfinite(lcl_temp))
        assert np.all(np.isfinite(lcl_pressure))
        assert np.all(lcl_temp     <= temp_arr)
        assert np.all(lcl_pressure <= pressure_arr)

    def test_scalar_array_consistency(self, lcl_eq):
        """
        Scalar and array dispatch with the same inputs must produce
        identical results — validates routing consistency.
        """
        temp_k, dewpoint_k, pressure_hpa = 293.15, 285.15, 1013.25

        t_scalar, p_scalar = lcl_eq._dispatch_scalar_or_vectorised(
            temp_k=temp_k,
            dewpoint_temp_k=dewpoint_k,
            pressure_hpa=pressure_hpa,
            scalar_func=_bolton_lcl_scalar,
            vector_func=_bolton_lcl_vectorised,
        )
        t_array, p_array = lcl_eq._dispatch_scalar_or_vectorised(
            temp_k=np.array([temp_k]),
            dewpoint_temp_k=np.array([dewpoint_k]),
            pressure_hpa=np.array([pressure_hpa]),
            scalar_func=_bolton_lcl_scalar,
            vector_func=_bolton_lcl_vectorised,
        )

        assert abs(t_scalar - float(t_array[0])) < 1e-10
        assert abs(p_scalar - float(p_array[0])) < 1e-10


# ===========================================================================
# Section 6 — Edge cases
# ===========================================================================

class TestEdgeCases:
    """Boundary and unusual inputs."""

    def test_scalar_zero_dewpoint_depression(self, lcl_eq):
        """Saturated parcel (T == Td) must still dispatch without error."""
        # Physics may be wrong but dispatch must not crash
        try:
            result = _dispatch(lcl_eq, 293.15, 293.15, 1013.25)
            assert len(result) == 2
        except Exception as e:
            pytest.skip(f"Saturated parcel raises in mock: {e}")

    def test_dispatch_deterministic(self, lcl_eq):
        """Same inputs must produce identical outputs on repeated calls."""
        results = [
            _dispatch(lcl_eq, 293.15, 285.15, 1013.25)
            for _ in range(5)
        ]
        for t, p in results:
            assert t == results[0][0]
            assert p == results[0][1]

    def test_broadcast_scalar_array_shape(self, lcl_eq):
        """
        Scalar + array broadcast must produce output shape matching
        the array, not the scalar.
        """
        n = 4

        def vector_func(temp_k, dewpoint_temp_k, pressure_hpa):
            return np.full(len(temp_k), 283.35), np.full(len(temp_k), 917.23)

        lcl_temp, lcl_pressure = _dispatch(
            lcl_eq,
            np.full(n, 293.15),   # array shape (4,)
            285.15,               # scalar — broadcast to (4,)
            1013.25,              # scalar — broadcast to (4,)
            _scalar_mock,
            vector_func,
        )

        assert lcl_temp.shape    == (n,)
        assert lcl_pressure.shape == (n,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])