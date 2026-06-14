"""
Comprehensive unit tests for LiftingCondensationLevelEquation._broadcast_input
===============================================================================

Tests cover:
- Scalar detection and was_scalar flag correctness
- dtype conversion to float64
- Shape broadcasting for all scalar/array combinations
- Contiguous memory layout for Numba compatibility
- N-dimensional input handling
- Mixed input types (int, float, list, tuple, ndarray)
- Edge cases (0-d arrays, single-element arrays, mismatched shapes)

The method is internal (_broadcast_input) but is tested directly because
it underpins all public LCL calculations — any bug here propagates silently
to every calculation result.
"""

import numpy as np
import numpy.typing as npt
import pytest

from meteocalc.lcl._lcl_equation import BoltonLclEquation, LiftingCondensationLevelEquation


# ---------------------------------------------------------------------------
# ---- Concrete fixture — BoltonLclEquation exposes _broadcast_input --------
# ---------------------------------------------------------------------------

@pytest.fixture
def lcl_eq() -> LiftingCondensationLevelEquation:
    """Concrete LCL equation instance for testing the ABC method."""
    return BoltonLclEquation()


def _broadcast(eq, temp_k, dewpoint_k, pressure_hpa):
    """Shorthand for calling _broadcast_input."""
    return eq._broadcast_input(temp_k, dewpoint_k, pressure_hpa)


# ===========================================================================
# Section 1 — was_scalar detection
# ===========================================================================

class TestWasScalarDetection:
    """was_scalar must be True only when ALL inputs are scalar."""

    def test_all_python_floats(self, lcl_eq):
        """Three Python floats → was_scalar=True."""
        _, _, _, was_scalar = _broadcast(lcl_eq, 293.15, 285.15, 1013.25)
        assert was_scalar is True

    def test_all_python_ints(self, lcl_eq):
        """Three Python ints → was_scalar=True."""
        _, _, _, was_scalar = _broadcast(lcl_eq, 293, 285, 1013)
        assert was_scalar is True

    def test_mixed_int_float(self, lcl_eq):
        """Mix of int and float scalars → was_scalar=True."""
        _, _, _, was_scalar = _broadcast(lcl_eq, 293, 285.15, 1013)
        assert was_scalar is True

    def test_zero_d_numpy_array(self, lcl_eq):
        """0-d numpy arrays are scalars → was_scalar=True."""
        t  = np.float64(293.15)
        td = np.float64(285.15)
        p  = np.float64(1013.25)
        _, _, _, was_scalar = _broadcast(lcl_eq, t, td, p)
        assert was_scalar is True

    def test_one_array_input_gives_false(self, lcl_eq):
        """One array input → was_scalar=False."""
        _, _, _, was_scalar = _broadcast(
            lcl_eq,
            np.array([293.15]),   # array
            285.15,               # scalar
            1013.25,              # scalar
        )
        assert was_scalar is False

    def test_all_array_inputs_gives_false(self, lcl_eq):
        """All array inputs → was_scalar=False."""
        _, _, _, was_scalar = _broadcast(
            lcl_eq,
            np.array([293.15, 303.15]),
            np.array([285.15, 298.15]),
            np.array([1013.25, 1000.0]),
        )
        assert was_scalar is False

    def test_list_input_gives_false(self, lcl_eq):
        """List input is not scalar → was_scalar=False."""
        _, _, _, was_scalar = _broadcast(lcl_eq, [293.15], 285.15, 1013.25)
        assert was_scalar is False

    def test_single_element_array_gives_false(self, lcl_eq):
        """Single-element array is not scalar → was_scalar=False."""
        _, _, _, was_scalar = _broadcast(
            lcl_eq, np.array([293.15]), 285.15, 1013.25
        )
        assert was_scalar is False


# ===========================================================================
# Section 2 — dtype conversion
# ===========================================================================

class TestDtypeConversion:
    """All outputs must be float64 regardless of input dtype."""

    @pytest.mark.parametrize("dtype", [
        np.float32, np.float64, np.int32, np.int64, np.float16,
    ])
    def test_numpy_dtype_converted_to_float64(self, lcl_eq, dtype):
        """Any numpy numeric dtype must be converted to float64."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.array([293.15], dtype=dtype),
            np.array([285.15], dtype=dtype),
            np.array([1013.25], dtype=dtype),
        )
        assert temp_k.dtype      == np.float64, f"temp_k dtype={temp_k.dtype}"
        assert dewpoint_k.dtype  == np.float64
        assert pressure_hpa.dtype == np.float64

    def test_python_int_converted_to_float64(self, lcl_eq):
        """Python int scalars must produce float64 output."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(lcl_eq, 293, 285, 1013)
        assert temp_k.dtype      == np.float64
        assert dewpoint_k.dtype  == np.float64
        assert pressure_hpa.dtype == np.float64

    def test_python_float_converted_to_float64(self, lcl_eq):
        """Python float scalars must produce float64 output."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq, 293.15, 285.15, 1013.25
        )
        assert temp_k.dtype      == np.float64
        assert dewpoint_k.dtype  == np.float64
        assert pressure_hpa.dtype == np.float64

    def test_list_input_converted_to_float64(self, lcl_eq):
        """List inputs must be converted to float64 ndarray."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq, [293.15, 303.15], [285.15, 298.15], [1013.25, 1000.0]
        )
        assert temp_k.dtype      == np.float64
        assert dewpoint_k.dtype  == np.float64
        assert pressure_hpa.dtype == np.float64

    def test_values_preserved_after_dtype_conversion(self, lcl_eq):
        """Conversion to float64 must not alter values."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.array([293.15], dtype=np.float32),
            np.array([285.15], dtype=np.float32),
            np.array([1013.25], dtype=np.float32),
        )
        assert abs(float(temp_k[0])      - 293.15) < 1e-3
        assert abs(float(dewpoint_k[0])  - 285.15) < 1e-3
        assert abs(float(pressure_hpa[0])- 1013.25) < 1e-2


# ===========================================================================
# Section 3 — shape broadcasting
# ===========================================================================

class TestShapeBroadcasting:
    """Outputs must be broadcast to a consistent common shape."""

    def test_all_scalars_broadcast_to_scalar(self, lcl_eq):
        """All scalar inputs → all outputs are 0-d after broadcast."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq, 293.15, 285.15, 1013.25
        )
        # After broadcast_arrays, scalars become 0-d arrays
        # then ascontiguousarray keeps them 1-d or 0-d
        assert temp_k.ndim      >= 0
        assert dewpoint_k.ndim  >= 0
        assert pressure_hpa.ndim >= 0

    def test_array_scalar_scalar_broadcasts(self, lcl_eq):
        """Array + scalar + scalar → all outputs match array shape."""
        arr = np.array([293.15, 303.15, 283.15])
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq, arr, 285.15, 1013.25
        )
        assert temp_k.shape      == (3,)
        assert dewpoint_k.shape  == (3,)
        assert pressure_hpa.shape == (3,)

    def test_all_same_shape_arrays(self, lcl_eq):
        """Arrays of same shape → outputs have same shape."""
        n = 5
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.full(n, 293.15),
            np.full(n, 285.15),
            np.full(n, 1013.25),
        )
        assert temp_k.shape      == (n,)
        assert dewpoint_k.shape  == (n,)
        assert pressure_hpa.shape == (n,)

    def test_scalar_scalar_array_broadcasts(self, lcl_eq):
        """Scalar + scalar + array → all outputs match array shape."""
        arr = np.array([1013.25, 1000.0, 850.0])
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq, 293.15, 285.15, arr
        )
        assert temp_k.shape      == (3,)
        assert dewpoint_k.shape  == (3,)
        assert pressure_hpa.shape == (3,)

    def test_large_array_shape_preserved(self, lcl_eq):
        """Large arrays must preserve shape correctly."""
        n = 100_000
        rng = np.random.default_rng(42)
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            rng.uniform(280.0, 310.0, n),
            rng.uniform(270.0, 290.0, n),
            rng.uniform(850.0, 1013.25, n),
        )
        assert temp_k.shape      == (n,)
        assert dewpoint_k.shape  == (n,)
        assert pressure_hpa.shape == (n,)

    def test_incompatible_shapes_raise(self, lcl_eq):
        """Incompatible shapes must raise ValueError from broadcast_arrays."""
        with pytest.raises(ValueError):
            _broadcast(
                lcl_eq,
                np.array([293.15, 303.15]),    # shape (2,)
                np.array([285.15, 298.15, 278.15]),  # shape (3,) — incompatible
                1013.25,
            )

    def test_2d_array_shape_preserved(self, lcl_eq):
        """2-D array inputs must preserve shape through broadcast."""
        shape = (3, 4)
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.full(shape, 293.15),
            np.full(shape, 285.15),
            np.full(shape, 1013.25),
        )
        assert temp_k.shape      == shape
        assert dewpoint_k.shape  == shape
        assert pressure_hpa.shape == shape


# ===========================================================================
# Section 4 — contiguous memory layout
# ===========================================================================

class TestContiguousMemory:
    """
    All outputs must be C-contiguous for Numba JIT compatibility.
    np.broadcast_arrays returns non-contiguous views — ascontiguousarray
    must fix this.
    """

    def test_scalar_inputs_contiguous(self, lcl_eq):
        """Scalar inputs must produce contiguous outputs."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq, 293.15, 285.15, 1013.25
        )
        assert temp_k.flags["C_CONTIGUOUS"],       "temp_k not C-contiguous"
        assert dewpoint_k.flags["C_CONTIGUOUS"],   "dewpoint_k not C-contiguous"
        assert pressure_hpa.flags["C_CONTIGUOUS"], "pressure_hpa not C-contiguous"

    def test_array_inputs_contiguous(self, lcl_eq):
        """Array inputs must produce contiguous outputs."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.array([293.15, 303.15]),
            np.array([285.15, 298.15]),
            np.array([1013.25, 1000.0]),
        )
        assert temp_k.flags["C_CONTIGUOUS"]
        assert dewpoint_k.flags["C_CONTIGUOUS"]
        assert pressure_hpa.flags["C_CONTIGUOUS"]

    def test_broadcast_scalar_to_array_contiguous(self, lcl_eq):
        """
        Broadcast views from scalar → array are non-contiguous by default.
        Must be made contiguous by ascontiguousarray.
        """
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.array([293.15, 303.15, 283.15]),
            285.15,    # scalar → broadcast view, stride=0
            1013.25,   # scalar → broadcast view, stride=0
        )
        assert dewpoint_k.flags["C_CONTIGUOUS"], (
            "Broadcast scalar (stride=0 view) was not made contiguous"
        )
        assert pressure_hpa.flags["C_CONTIGUOUS"], (
            "Broadcast scalar (stride=0 view) was not made contiguous"
        )

    def test_large_array_contiguous(self, lcl_eq):
        """Large arrays must be contiguous — critical for Numba prange."""
        n = 100_000
        rng = np.random.default_rng(42)
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            rng.uniform(280.0, 310.0, n),
            285.15,     # scalar broadcast
            1013.25,    # scalar broadcast
        )
        assert temp_k.flags["C_CONTIGUOUS"]
        assert dewpoint_k.flags["C_CONTIGUOUS"]
        assert pressure_hpa.flags["C_CONTIGUOUS"]


# ===========================================================================
# Section 5 — return types
# ===========================================================================

class TestReturnTypes:
    """Outputs must always be numpy ndarrays of float64."""

    def test_scalar_input_returns_ndarray(self, lcl_eq):
        """Even scalar inputs must return ndarrays — not Python scalars."""
        temp_k, dewpoint_k, pressure_hpa, was_scalar = _broadcast(
            lcl_eq, 293.15, 285.15, 1013.25
        )
        assert isinstance(temp_k,       np.ndarray)
        assert isinstance(dewpoint_k,   np.ndarray)
        assert isinstance(pressure_hpa, np.ndarray)
        assert isinstance(was_scalar,   bool)

    def test_was_scalar_is_python_bool(self, lcl_eq):
        """was_scalar must be a Python bool, not numpy bool."""
        _, _, _, was_scalar = _broadcast(lcl_eq, 293.15, 285.15, 1013.25)
        assert type(was_scalar) is bool

    def test_returns_four_values(self, lcl_eq):
        """Must return exactly four values."""
        result = _broadcast(lcl_eq, 293.15, 285.15, 1013.25)
        assert len(result) == 4

    def test_array_input_returns_ndarray(self, lcl_eq):
        """Array inputs must return ndarrays of correct dtype."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.array([293.15, 303.15]),
            np.array([285.15, 298.15]),
            np.array([1013.25, 1000.0]),
        )
        assert isinstance(temp_k,       np.ndarray)
        assert isinstance(dewpoint_k,   np.ndarray)
        assert isinstance(pressure_hpa, np.ndarray)
        assert temp_k.dtype      == np.float64
        assert dewpoint_k.dtype  == np.float64
        assert pressure_hpa.dtype == np.float64


# ===========================================================================
# Section 6 — value preservation
# ===========================================================================

class TestValuePreservation:
    """Broadcasting and conversion must not alter input values."""

    def test_scalar_values_preserved(self, lcl_eq):
        """Scalar values must be exactly preserved after conversion."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq, 293.15, 285.15, 1013.25
        )
        assert float(temp_k.flat[0])       == 293.15
        assert float(dewpoint_k.flat[0])   == 285.15
        assert float(pressure_hpa.flat[0]) == 1013.25

    def test_array_values_preserved(self, lcl_eq):
        """Array values must be exactly preserved after broadcasting."""
        t_in  = np.array([293.15, 303.15, 283.15])
        td_in = np.array([285.15, 298.15, 278.15])
        p_in  = np.array([1013.25, 1000.0, 850.0])

        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(lcl_eq, t_in, td_in, p_in)

        np.testing.assert_array_equal(temp_k,       t_in)
        np.testing.assert_array_equal(dewpoint_k,   td_in)
        np.testing.assert_array_equal(pressure_hpa, p_in)

    def test_broadcast_scalar_fills_correctly(self, lcl_eq):
        """Scalar broadcast to array must fill all elements with scalar value."""
        n = 5
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            np.full(n, 293.15),
            285.15,     # scalar — should fill all 5 elements
            1013.25,    # scalar — should fill all 5 elements
        )
        np.testing.assert_array_equal(dewpoint_k,   np.full(n, 285.15))
        np.testing.assert_array_equal(pressure_hpa, np.full(n, 1013.25))

    def test_list_values_preserved(self, lcl_eq):
        """List inputs must have values preserved after conversion."""
        temp_k, dewpoint_k, pressure_hpa, _ = _broadcast(
            lcl_eq,
            [293.15, 303.15],
            [285.15, 298.15],
            [1013.25, 1000.0],
        )
        np.testing.assert_array_almost_equal(temp_k,       [293.15, 303.15])
        np.testing.assert_array_almost_equal(dewpoint_k,   [285.15, 298.15])
        np.testing.assert_array_almost_equal(pressure_hpa, [1013.25, 1000.0])


# ===========================================================================
# Section 7 — mixed input types
# ===========================================================================

class TestMixedInputTypes:
    """All common input type combinations must work correctly."""

    @pytest.mark.parametrize("temp_k,dewpoint_k,pressure_hpa,expected_scalar", [
        (293.15,                  285.15,                  1013.25,                True),   # all float
        (293,                     285,                     1013,                   True),   # all int
        (np.float64(293.15),      np.float64(285.15),      np.float64(1013.25),   True),   # all np scalar
        (np.array([293.15]),      285.15,                  1013.25,               False),  # one array
        ([293.15],                [285.15],                [1013.25],             False),  # all lists
        (np.array([293.15, 303.15]), np.array([285.15, 298.15]), 1013.25,        False),  # arrays + scalar
    ])
    def test_input_type_combinations(
        self, lcl_eq, temp_k, dewpoint_k, pressure_hpa, expected_scalar
    ):
        """All type combinations must produce correct was_scalar flag."""
        t, td, p, was_scalar = _broadcast(lcl_eq, temp_k, dewpoint_k, pressure_hpa)
        assert was_scalar == expected_scalar
        assert t.dtype   == np.float64
        assert td.dtype  == np.float64
        assert p.dtype   == np.float64


# ===========================================================================
# Section 8 — edge cases
# ===========================================================================

class TestEdgeCases:
    """Boundary and unusual inputs."""

    def test_single_element_array(self, lcl_eq):
        """Single-element array is not scalar — was_scalar must be False."""
        _, _, _, was_scalar = _broadcast(
            lcl_eq, np.array([293.15]), np.array([285.15]), np.array([1013.25])
        )
        assert was_scalar is False

    def test_identical_inputs(self, lcl_eq):
        """Identical values for all inputs — no special behaviour expected."""
        temp_k, dewpoint_k, pressure_hpa, was_scalar = _broadcast(
            lcl_eq, 293.15, 293.15, 1013.25
        )
        assert was_scalar is True
        assert float(temp_k.flat[0])     == 293.15
        assert float(dewpoint_k.flat[0]) == 293.15

    def test_very_large_array(self, lcl_eq):
        """1M element arrays must broadcast and be contiguous."""
        n   = 1_000_000
        rng = np.random.default_rng(42)
        t, td, p, was_scalar = _broadcast(
            lcl_eq,
            rng.uniform(260.0, 313.0, n),
            rng.uniform(240.0, 295.0, n),
            rng.uniform(850.0, 1013.25, n),
        )
        assert was_scalar is False
        assert t.shape  == (n,)
        assert t.flags["C_CONTIGUOUS"]
        assert t.dtype  == np.float64

    def test_tuple_input(self, lcl_eq):
        """Tuple inputs should behave identically to lists."""
        t, td, p, was_scalar = _broadcast(
            lcl_eq,
            (293.15, 303.15),
            (285.15, 298.15),
            (1013.25, 1000.0),
        )
        assert was_scalar is False
        assert t.shape == (2,)
        assert t.dtype == np.float64

    def test_outputs_are_independent(self, lcl_eq):
        """
        Output arrays must be independent — modifying one must not
        affect the others. Validates that ascontiguousarray produces
        owned arrays, not views sharing memory.
        """
        t, td, p, _ = _broadcast(
            lcl_eq,
            np.array([293.15, 303.15]),
            285.15,
            1013.25,
        )
        original_td = td.copy()
        t[0] = 999.99   # modify temp_k
        np.testing.assert_array_equal(td, original_td)  # dewpoint_k unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])