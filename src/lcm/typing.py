from enum import Enum
from typing import Any

from jax import Array

# Many JAX functions are designed to work with scalar numerical values. This also
# includes zero dimensional jax arrays.
Scalar = int | float | Array

ParamsDict = dict[str, Any]


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None
