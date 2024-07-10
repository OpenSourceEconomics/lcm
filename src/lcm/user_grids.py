import dataclasses as dc
from collections.abc import Iterable
from typing import Self, get_args

from lcm.typing import ScalarUserInput


class Grid:
    """LCM Grid base class."""


@dc.dataclass(frozen=True)
class DiscreteGrid(Grid):
    """A grid of discrete values.

    Attributes:
        options: The options in the grid. Must be an iterable of scalar int or float
            values.

    """

    options: Iterable[ScalarUserInput]

    def __post_init__(self) -> None:
        if not isinstance(self.options, Iterable):
            raise LcmGridInitializationError(
                "options must be an iterable of scalar int or float values",
            )

        errors = _validate_discrete_grid(self.options)
        if errors:
            raise LcmGridInitializationError(_format_errors(errors))

    def replace(self, options: Iterable[ScalarUserInput]) -> "DiscreteGrid":
        """Replace the grid with new values.

        Args:
            options: The new options in the grid.

        Returns:
            The updated grid.

        """
        return dc.replace(self, options=options)


@dc.dataclass(frozen=True)
class ContinuousGrid(Grid):
    """LCM Continuous Grid base class."""

    start: ScalarUserInput
    stop: ScalarUserInput
    n_points: int

    def __post_init__(self) -> None:
        errors = _validate_continuous_grid(
            start=self.start,
            stop=self.stop,
            n_points=self.n_points,
        )
        if errors:
            raise LcmGridInitializationError(_format_errors(errors))

    def replace(self, **kwargs) -> Self:
        """Replace the grid with new values.

        Args:
            **kwargs:
                - start: The new start value of the grid.
                - stop: The new stop value of the grid.
                - n_points: The new number of points in the grid.

        Returns:
            The updated grid.


        """
        return dc.replace(self, **kwargs)


class LinspaceGrid(ContinuousGrid):
    """A linear grid of continuous values.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """


class LogspaceGrid(ContinuousGrid):
    """A logarithmic grid of continuous values.

    Attributes:
        start: The start value of the grid. Must be a scalar int or float value.
        stop: The stop value of the grid. Must be a scalar int or float value.
        n_points: The number of points in the grid. Must be an int greater than 0.

    """


# ======================================================================================
# Validate user input
# ======================================================================================


class LcmGridInitializationError(Exception):
    """Raised when there is an error in the grid initialization."""


def _format_errors(errors: list[str]) -> str:
    """Convert list of error messages into a single string.

    If list is empty, returns the empty string.

    """
    if len(errors) == 0:
        formatted = ""
    elif len(errors) == 1:
        formatted = errors[0]
    else:
        enumerated = "\n\n".join([f"{i}. {error}" for i, error in enumerate(errors, 1)])
        formatted = f"The following errors occurred:\n\n{enumerated}"
    return formatted


# Discrete grid
# ======================================================================================


def _validate_discrete_grid(options: list[ScalarUserInput]) -> list[str]:
    """Validate the discrete grid options.

    Args:
        options: The user options to validate.

    Returns:
        list[str]: A list of error messages.

    """
    error_messages = []

    if not len(options) > 0:
        error_messages.append("options must contain at least one element")

    if not all(isinstance(option, get_args(ScalarUserInput)) for option in options):
        error_messages.append("options must contain only scalar int or float values")

    if len(options) != len(set(options)):
        error_messages.append("options must contain unique values")

    return error_messages


# Continuous grid
# ======================================================================================


def _validate_continuous_grid(
    start: ScalarUserInput,
    stop: ScalarUserInput,
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

    if not (valid_start_type := isinstance(start, get_args(ScalarUserInput))):
        error_messages.append("start must be a scalar int or float value")

    if not (valid_stop_type := isinstance(stop, get_args(ScalarUserInput))):
        error_messages.append("stop must be a scalar int or float value")

    if not isinstance(n_points, int) or n_points < 1:
        error_messages.append(
            f"n_points must be an int greater than 0 but is {n_points}",
        )

    if valid_start_type and valid_stop_type and start >= stop:
        error_messages.append("start must be less than stop")

    return error_messages
