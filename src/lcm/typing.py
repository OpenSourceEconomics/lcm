from enum import Enum
from typing import Any, Protocol, Self

from jax import Array

# Many JAX functions are designed to work with scalar numerical values. This also
# includes zero dimensional jax arrays.
Scalar = int | float | Array


ParamsDict = dict[str, Any]


class UserFunction(Protocol):
    """A function provided by the user. Only used for type checking."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D102


class InternalUserFunction(Protocol):
    """A function provided by the user. Only used for type checking."""

    def __call__(  # noqa: D102
        self, *args: Scalar, params: ParamsDict, **kwargs: Scalar
    ) -> Scalar: ...


class DiscreteProblemSolverFunction(Protocol):
    """The solution to the discrete problem. Only used for type checking."""

    def __call__(self, values: Array, params: ParamsDict) -> Array: ...  # noqa: D102


class ShockType(Enum):
    """Type of shocks."""

    EXTREME_VALUE = "extreme_value"
    NONE = None


class Target(Enum):
    """Target of the function."""

    SOLVE = "solve"
    SIMULATE = "simulate"


class ReplaceMixin:
    """Mixin for replacing init-attributes of a class."""

    def replace(self: Self, **kwargs: Any) -> Self:  # noqa: ANN401
        """Replace the init-attributes of the class with the given values."""
        return type(self)(**{**self.__dict__, **kwargs})
