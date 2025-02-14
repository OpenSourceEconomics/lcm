from enum import Enum
from typing import Any, Protocol

from jax import Array

# Many JAX functions are designed to work with scalar numerical values. This also
# includes zero dimensional jax arrays.
Scalar = int | float | Array


class UserFunction(Protocol):
    """A function that can be provided by the user. Only used for type checking."""

    def __call__(self, *args: Scalar, **kwargs: Scalar) -> Scalar: ...  # noqa: D102


ParamsDict = dict[str, Any]


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None
