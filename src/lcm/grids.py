"""Collection of classes that are used by the user to define the model and grids."""

import dataclasses as dc
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import NotRequired, TypedDict, cast

import jax.numpy as jnp

from lcm.exceptions import GridInitializationError, format_messages
from lcm.grid_helpers import linspace, logspace
from lcm.interfaces import ContinuousGridInfo
from lcm.typing import ContinuousGridType

build_grid_mapping = {
    "linspace": linspace,
    "logspace": logspace,
}


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

        if errors := _validate_discrete_grid(self.options):
            msg = format_messages(errors)
            raise GridInitializationError(msg)

    def to_jax(self) -> jnp.ndarray:
        """Convert the grid to a Jax array."""
        return jnp.array(list(self.options))

    def replace(self, options: Collection[int | float]) -> "DiscreteGrid":
        """Replace the grid with new values.

        Args:
            options: The new options in the grid.

        Returns:
            The updated grid.

        """
        return dc.replace(self, options=options)


@dataclass(frozen=True)
class ContinuousGrid(Grid):
    """LCM Continuous Grid base class."""

    kind: ContinuousGridType = field(init=False, default=None)  # type: ignore[arg-type]
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

    @property
    def info(self) -> ContinuousGridInfo:
        """Get the grid info."""
        return ContinuousGridInfo(
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )

    def to_jax(self) -> jnp.ndarray:
        """Convert the grid to a Jax array."""
        return build_grid_mapping[self.kind](
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )


class ContinuousGridReplacements(TypedDict):
    """Dictionary of arguments that can be replaced using the `replace` method."""

    start: NotRequired[int | float]
    stop: NotRequired[int | float]
    n_points: NotRequired[int]


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

    kind: ContinuousGridType = "linspace"

    def replace(
        self,
        start: float | None = None,
        stop: float | None = None,
        n_points: int | None = None,
    ) -> "LinspaceGrid":
        """Replace the grid with new values.

        Args:
            start: The new start value of the grid.
            stop: The new stop value of the grid.
            n_points: The new number of points in the grid.

        Returns:
            The updated grid.

        """
        replacements = {"start": start, "stop": stop, "n_points": n_points}
        replacements = {k: v for k, v in replacements.items() if v is not None}
        kwargs = cast(ContinuousGridReplacements, replacements)
        return dc.replace(self, **kwargs)


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

    kind: ContinuousGridType = "logspace"

    def replace(
        self,
        start: float | None = None,
        stop: float | None = None,
        n_points: int | None = None,
    ) -> "LogspaceGrid":
        """Replace the grid with new values.

        Args:
            start: The new start value of the grid.
            stop: The new stop value of the grid.
            n_points: The new number of points in the grid.

        Returns:
            The updated grid.

        """
        replacements = {"start": start, "stop": stop, "n_points": n_points}
        replacements = {k: v for k, v in replacements.items() if v is not None}
        kwargs = cast(ContinuousGridReplacements, replacements)
        return dc.replace(self, **kwargs)


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

    if list(options) != list(range(len(options))):
        error_messages.append(
            "options must be a list of consecutive integers starting from 0",
        )

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

    if not (valid_start_type := isinstance(start, int | float)):
        error_messages.append("start must be a scalar int or float value")

    if not (valid_stop_type := isinstance(stop, int | float)):
        error_messages.append("stop must be a scalar int or float value")

    if not isinstance(n_points, int) or n_points < 1:
        error_messages.append(
            f"n_points must be an int greater than 0 but is {n_points}",
        )

    if valid_start_type and valid_stop_type and start >= stop:
        error_messages.append("start must be less than stop")

    return error_messages
