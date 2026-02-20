"""
reusable tools for validating and checking enums.

Author: Cian Quezon
"""

from enum import Enum
from typing import Type, TypeVar, Union

E = TypeVar("E", bound=Enum)


def parse_enum(value: Union[str, E], enum_class: Type[E]) -> E:
    """Parse and validate enum from string or enum instance.

    Converts string values to enum instances with case-insensitive matching.
    If an enum instance is provided, validates it belongs to the expected class.

    Parameters
    ----------
    value : str or E
        The value to parse. Can be either:
        - A string matching an enum value (case-insensitive)
        - An instance of the target enum class
    enum_class : Type[E]
        The enum class to parse into.

    Returns
    -------
    E
        Valid instance of the specified enum class.

    Raises
    ------
    ValueError
        If the string value doesn't match any enum member.
    TypeError
        If value is neither a string nor an instance of the enum class.

    Examples
    --------
    >>> from enum import Enum
    >>> class Color(Enum):
    ...     RED = "red"
    ...     BLUE = "blue"

    >>> # Parse from string (case-insensitive)
    >>> parse_enum("RED", Color)
    <Color.RED: 'red'>

    >>> parse_enum("blue", Color)
    <Color.BLUE: 'blue'>

    >>> # Pass through existing enum
    >>> parse_enum(Color.RED, Color)
    <Color.RED: 'red'>

    >>> # Invalid string raises ValueError
    >>> parse_enum("green", Color)
    ValueError: Invalid enum 'green'. Available are the following: [red, blue]

    >>> # Invalid type raises TypeError
    >>> parse_enum(123, Color)
    TypeError: value must be str or Color, got int
    """
    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        try:
            final_enum = enum_class(value.lower())
            return final_enum

        except ValueError as err:
            valid_enums = ", ".join([e.value for e in enum_class])
            raise ValueError(
                f"Invalid enum '{value}'. Available are the following: [{valid_enums}]"
            ) from err

    raise TypeError(
        f"value must be str or {enum_class.__name__}, got {type(value).__name__}"
    )
