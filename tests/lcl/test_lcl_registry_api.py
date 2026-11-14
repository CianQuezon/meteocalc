"""
Tests for Lcl.get_equations_available and Lcl.get_equation — registry API
=========================================================================

Both methods expose the LCL equation registry to callers.
``get_equations_available`` returns the list of registered equation names;
``get_equation`` resolves a name (string or enum) to its equation class.

Test sections
-------------
1.  get_equations_available — return type, contents, completeness, stability
2.  get_equation — string input, enum input, return type, instantiability,
                   invalid-name errors, end-to-end calculation
"""

import pytest
import numpy as np

from meteocalc.lcl._enums import LclEquationName
from meteocalc.lcl._lcl_equation import (
    BoltonLclEquation,
    IterativeLclEquation,
    LiftingCondensationLevelEquation,
)
from meteocalc.lcl._types import LCL_EQUATION_MAP
from meteocalc.lcl.core import Lcl
from meteocalc.shared.constants import eps
from meteocalc.vapor.core import Vapor


# ===========================================================================
# Section 1 — Lcl.get_equations_available
# ===========================================================================

class TestGetEquationsAvailable:
    """
    ``get_equations_available()`` must return a complete, correctly-typed list
    of every registered equation name.
    """

    def test_returns_list(self):
        assert isinstance(Lcl.get_equations_available(), list)

    def test_contains_bolton(self):
        assert "bolton" in Lcl.get_equations_available()

    def test_contains_iterative(self):
        assert "iterative" in Lcl.get_equations_available()

    def test_length_matches_enum(self):
        """Length must equal the number of LclEquationName members."""
        assert len(Lcl.get_equations_available()) == len(LclEquationName)

    def test_all_entries_are_strings(self):
        for name in Lcl.get_equations_available():
            assert isinstance(name, str), f"Expected str, got {type(name)}: {name!r}"

    def test_all_entries_are_non_empty(self):
        for name in Lcl.get_equations_available():
            assert len(name) > 0, "Empty string in equation list"

    def test_each_name_parseable_as_enum(self):
        """Every returned name must round-trip through ``LclEquationName``."""
        from meteocalc.shared._enum_tools import parse_enum
        for name in Lcl.get_equations_available():
            enum_val = parse_enum(name, LclEquationName)
            assert isinstance(enum_val, LclEquationName), (
                f"'{name}' did not parse to LclEquationName"
            )

    def test_each_name_registered_in_equation_map(self):
        """Every returned name must have an entry in ``LCL_EQUATION_MAP``."""
        from meteocalc.shared._enum_tools import parse_enum
        for name in Lcl.get_equations_available():
            key = parse_enum(name, LclEquationName)
            assert key in LCL_EQUATION_MAP, (
                f"'{name}' not found in LCL_EQUATION_MAP"
            )

    def test_no_duplicates(self):
        names = Lcl.get_equations_available()
        assert len(names) == len(set(names)), "Duplicate equation names returned"

    def test_stable_across_calls(self):
        """Repeated calls must return identical lists."""
        first  = Lcl.get_equations_available()
        second = Lcl.get_equations_available()
        assert first == second

    def test_order_matches_enum_definition(self):
        """Order must match iteration order of ``LclEquationName``."""
        expected = [e.value for e in LclEquationName]
        assert Lcl.get_equations_available() == expected


# ===========================================================================
# Section 2 — Lcl.get_equation
# ===========================================================================

class TestGetEquation:
    """
    ``get_equation(name)`` must resolve any valid name to the correct equation
    class, accept both string and enum inputs, and raise on unknown names.
    """

    # ---- Return type -------------------------------------------------------

    def test_returns_class_not_instance(self):
        cls = Lcl.get_equation("bolton")
        assert isinstance(cls, type), f"Expected a class, got {type(cls)}"

    def test_bolton_string_returns_bolton_class(self):
        assert Lcl.get_equation("bolton") is BoltonLclEquation

    def test_iterative_string_returns_iterative_class(self):
        assert Lcl.get_equation("iterative") is IterativeLclEquation

    # ---- Enum input --------------------------------------------------------

    def test_bolton_enum_returns_bolton_class(self):
        assert Lcl.get_equation(LclEquationName.BOLTON) is BoltonLclEquation

    def test_iterative_enum_returns_iterative_class(self):
        assert Lcl.get_equation(LclEquationName.ITERATIVE) is IterativeLclEquation

    def test_string_and_enum_return_same_class_bolton(self):
        assert Lcl.get_equation("bolton") is Lcl.get_equation(LclEquationName.BOLTON)

    def test_string_and_enum_return_same_class_iterative(self):
        assert Lcl.get_equation("iterative") is Lcl.get_equation(LclEquationName.ITERATIVE)

    # ---- All registered equations ------------------------------------------

    def test_all_available_names_resolve(self):
        """Every name from ``get_equations_available`` must resolve without error."""
        for name in Lcl.get_equations_available():
            cls = Lcl.get_equation(name)
            assert cls is not None, f"get_equation('{name}') returned None"

    # ---- Subclass invariant ------------------------------------------------

    def test_bolton_class_is_subclass_of_base(self):
        assert issubclass(Lcl.get_equation("bolton"), LiftingCondensationLevelEquation)

    def test_iterative_class_is_subclass_of_base(self):
        assert issubclass(Lcl.get_equation("iterative"), LiftingCondensationLevelEquation)

    # ---- Instantiability and calculate interface ---------------------------

    def test_bolton_class_is_instantiable(self):
        cls      = Lcl.get_equation("bolton")
        instance = cls()
        assert isinstance(instance, BoltonLclEquation)

    def test_iterative_class_is_instantiable(self):
        cls      = Lcl.get_equation("iterative")
        instance = cls()
        assert isinstance(instance, IterativeLclEquation)

    def test_retrieved_bolton_class_can_calculate(self):
        """Instantiate and call ``calculate`` — must produce finite results."""
        cls           = Lcl.get_equation("bolton")
        eq            = cls()
        t_lcl, p_lcl  = eq.calculate(
            temp_k=293.15, dewpoint_temp_k=285.15, pressure_hpa=1013.25
        )
        assert isinstance(t_lcl, float)
        assert isinstance(p_lcl, float)
        import math
        assert math.isfinite(t_lcl)
        assert math.isfinite(p_lcl)
        assert t_lcl <= 293.15
        assert p_lcl <= 1013.25

    def test_retrieved_iterative_class_can_calculate(self):
        """Instantiate and call ``calculate`` with a vapour equation — must produce finite results."""
        cls       = Lcl.get_equation("iterative")
        eq        = cls()
        vapor_eq  = Vapor.get_equation("goff_gratch", phase="water")
        e         = float(vapor_eq.calculate(285.15))
        w         = eps * e / (1013.25 - e)
        t_lcl, p_lcl = eq.calculate(
            temp_k=293.15,
            dewpoint_temp_k=285.15,
            pressure_hpa=1013.25,
            mixing_ratio=w,
            vapor_equation=vapor_eq,
        )
        import math
        assert math.isfinite(t_lcl)
        assert math.isfinite(p_lcl)
        assert t_lcl <= 293.15

    def test_retrieved_class_matches_direct_import_bolton(self):
        """Class returned by ``get_equation`` must be the same object as direct import."""
        assert Lcl.get_equation("bolton") is BoltonLclEquation

    def test_retrieved_class_matches_direct_import_iterative(self):
        assert Lcl.get_equation("iterative") is IterativeLclEquation

    def test_stable_across_calls(self):
        """Repeated calls with same name must return the same class object."""
        assert Lcl.get_equation("bolton") is Lcl.get_equation("bolton")
        assert Lcl.get_equation("iterative") is Lcl.get_equation("iterative")

    # ---- Error handling ----------------------------------------------------

    def test_invalid_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            Lcl.get_equation("nonexistent")

    def test_empty_string_raises(self):
        with pytest.raises((ValueError, KeyError)):
            Lcl.get_equation("")

    def test_none_raises(self):
        with pytest.raises((ValueError, KeyError, AttributeError, TypeError)):
            Lcl.get_equation(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])