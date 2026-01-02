"""
reusable tools for validating and checking enums.

Author: Cian Quezon
"""

from enum import Enum
from typing import Type, TypeVar, Union

from meteorological_equations.vapor._enums import EquationName, SurfaceType
E = TypeVar("E", bound=Enum)

def parse_enum(value: Union[str, E], enum_class: Type[E]) -> E:
    """
    returns the enum based on the string input or enum type input

    Args:
        - value(Union[str, E]) = enum string or type
        - enum(Type[E]) = primary enum

    Returns:
        - Returns the enum selected
    """
    if isinstance(value, enum_class):
        return value
        
    if isinstance(value, str):
        try:
            final_enum = enum_class(value.lower())
            return final_enum
        
        except ValueError:
            valid_enums =', '.join([e.value for e in enum_class])
            raise ValueError(
                f"Invalid enum '{value}'. Available are the following: [{valid_enums}]"        
            )

    raise TypeError(
        f"value must be str or {enum_class.__name__}, got {type(value).__name__}"
    )


