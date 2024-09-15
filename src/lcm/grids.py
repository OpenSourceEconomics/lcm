"""Collection of classes that are used by the user to define the model and grids."""

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from lcm import grid_helpers
from lcm.exceptions import GridInitializationError, format_messages
from lcm.typing import Scalar


class Grid(ABC):
    """LCM Grid base class."""

    @abstractmethod
    def to_jax(self) -> jnp.ndarray:
        """Convert the grid to a Jax array."""


@dataclass(frozen=True)
class DiscreteGrid(Grid):
    """A grid of discrete values.

    Attributes:
        options: The options in the grid. Must be an iterable of scalar int or float
            values.

    """

    options: Collection[int | float]

    def __post_init__(self) -> None:
        if not isinstance(self.options, Collection):
            raise GridInitializationError(
                "options must be a collection of scalar int or float values, e.g., a ",
                "list or tuple",
            )

        errors = _validate_discrete_grid(self.options)
        if errors:
            msg = format_messages(errors)
            raise GridInitializationError(msg)

    def to_jax(self) -> Array:
        """Convert the grid to a Jax array."""
        return jnp.array(list(self.options))


@dataclass(frozen=True, kw_only=True)
class ContinuousGrid(Grid, ABC):
    """LCM Continuous Grid base class."""

    start: int | float
    stop: int | float
    n_points: int

    def __post_init__(self) -> None:
        errors = _validate_continuous_grid(
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )
        if errors:
            msg = format_messages(errors)
            raise GridInitializationError(msg)

    @abstractmethod
    def to_jax(self) -> Array:
        """Convert the grid to a Jax array."""

    @abstractmethod
    def get_coordinate(self, value: Scalar) -> Scalar:
        """Get the generalized coordinate of a value in the grid."""


class LinspaceGrid(ContinuousGrid):
    """A linear grid of continuous values.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 50.5, 100]`.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """

    def to_jax(self) -> Array:
        """Convert the grid to a Jax array."""
        return grid_helpers.linspace(self.start, self.stop, self.n_points)

    def get_coordinate(self, value: Scalar) -> Scalar:
        """Get the generalized coordinate of a value in the grid."""
        return grid_helpers.get_linspace_coordinate(
            value, self.start, self.stop, self.n_points
        )


class LogspaceGrid(ContinuousGrid):
    """A logarithmic grid of continuous values.

    Example:
    --------
    Let `start = 1`, `stop = 100`, and `n_points = 3`. The grid is `[1, 10, 100]`.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """

    def to_jax(self) -> Array:
        """Convert the grid to a Jax array."""
        return grid_helpers.logspace(self.start, self.stop, self.n_points)

    def get_coordinate(self, value: Scalar) -> Scalar:
        """Get the generalized coordinate of a value in the grid."""
        return grid_helpers.get_logspace_coordinate(
            value, self.start, self.stop, self.n_points
        )


# ======================================================================================
# Validate user input
# ======================================================================================


def _validate_discrete_grid(options: Collection[int | float]) -> list[str]:
    """Validate the discrete grid options.

    Args:
        options: The user options to validate.

    Returns:
        list[str]: A list of error messages.

    """
    error_messages = []

    if not len(options) > 0:
        error_messages.append("options must contain at least one element")

    if not all(isinstance(option, int | float) for option in options):
        error_messages.append("options must contain only scalar int or float values")

    if len(options) != len(set(options)):
        error_messages.append("options must contain unique values")

    return error_messages


def _validate_continuous_grid(
    start: float,
    stop: float,
    n_points: int,
) -> list[str]:
    """Validate the continuous grid parameters.

    Args:
        start: The start value of the grid.
        stop: The stop value of the grid.
        n_points: The number of points in the grid.

    Returns:
        list[str]: A list of error messages.

    """
    error_messages = []

    valid_start_type = isinstance(start, int | float)
    if not valid_start_type:
        error_messages.append("start must be a scalar int or float value")

    valid_stop_type = isinstance(stop, int | float)
    if not valid_stop_type:
        error_messages.append("stop must be a scalar int or float value")

    if not isinstance(n_points, int) or n_points < 1:
        error_messages.append(
            f"n_points must be an int greater than 0 but is {n_points}",
        )

    if valid_start_type and valid_stop_type and start >= stop:
        error_messages.append("start must be less than stop")

    return error_messages
