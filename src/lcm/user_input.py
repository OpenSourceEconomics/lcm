"""Collection of classes that are used by the user to define the model and grids."""

import dataclasses as dc
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import get_args

import jax.numpy as jnp

import lcm.grids as grids_module
from lcm.interfaces import ContinuousGridInfo
from lcm.typing import ContinuousGridType, ScalarUserInput


class Grid(ABC):
    """LCM Grid base class."""

    @abstractmethod
    def to_jax(self) -> jnp.ndarray:
        """Convert the grid to a Jax array."""


@dataclass(frozen=True)
class Model:
    """A user model which can be processed into an internal model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        functions: Dictionary of user provided functions that define the functional
            relationships between model variables. It must include at least a function
            called 'utility'.
        choices: Dictionary of user provided choices.
        states: Dictionary of user provided states.

    """

    description: str | None = None
    _: KW_ONLY
    n_periods: int
    functions: dict[str, Callable] = field(default_factory=dict)
    choices: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    _skip_checks: InitVar[bool] = False

    def __post_init__(self, _skip_checks: bool) -> None:
        if _skip_checks:
            return

        type_errors = _validate_model_attribute_types(self)
        if type_errors:
            raise LcmModelInitializationError(_format_errors(type_errors))

        logical_errors = _validate_logical_consistency_model(self)
        if logical_errors:
            raise LcmModelInitializationError(_format_errors(logical_errors))

    def replace(self, **kwargs) -> "Model":
        """Replace the attributes of the model.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the model.

        Returns:
            A new model with the replaced attributes.

        """
        return dc.replace(self, **kwargs)


@dataclass(frozen=True)
class DiscreteGrid(Grid):
    """A grid of discrete values.

    Attributes:
        options: The options in the grid. Must be an iterable of scalar int or float
            values.

    """

    options: Collection[ScalarUserInput]

    def __post_init__(self) -> None:
        if not isinstance(self.options, Collection):
            raise LcmGridInitializationError(
                "options must be a collection of scalar int or float values, e.g., a ",
                "list or tuple",
            )

        errors = _validate_discrete_grid(self.options)
        if errors:
            raise LcmGridInitializationError(_format_errors(errors))

    def to_jax(self) -> jnp.ndarray:
        """Convert the grid to a Jax array."""
        return jnp.asarray(list(self.options))

    def replace(self, options: Collection[ScalarUserInput]) -> "DiscreteGrid":
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
        space_func = getattr(grids_module, self.kind)
        return space_func(start=self.start, stop=self.stop, n_points=self.n_points)


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


    def replace(self, **kwargs) -> "LinspaceGrid":
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
    
    def replace(self, **kwargs) -> "LogspaceGrid":
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


# ======================================================================================
# Validate user input
# ======================================================================================


class LcmModelInitializationError(Exception):
    """Raised when there is an error in the model initialization."""


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


# Model
# ======================================================================================


def _validate_model_attribute_types(model: Model) -> list[str]:
    """Validate the types of the model attributes."""
    error_messages = []

    # Validate types of states and choices
    # ----------------------------------------------------------------------------------
    for attr_name in ("choices", "states"):
        attr = getattr(model, attr_name)
        if not isinstance(attr, dict):
            error_messages.append(f"{attr_name} must be a dictionary.")
        else:
            for k, v in attr.items():
                if not isinstance(k, str):
                    error_messages.append(f"{attr_name} key {k} must be a string.")
                if not isinstance(v, Grid):
                    error_messages.append(f"{attr_name} value {v} must be a LCM grid.")

    # Validate types of functions
    # ----------------------------------------------------------------------------------
    if not isinstance(model.functions, dict):
        error_messages.append("functions must be a dictionary.")
    else:
        for k, v in model.functions.items():
            if not isinstance(k, str):
                error_messages.append(f"functions key {k} must be a string.")
            if not callable(v):
                error_messages.append(f"functions value {v} must be a callable.")

    return error_messages


def _validate_logical_consistency_model(model: Model) -> list[str]:
    """Validate the logical consistency of the model."""
    error_messages = []

    if model.n_periods < 1:
        error_messages.append("Number of periods must be a positive integer.")

    if "utility" not in model.functions:
        error_messages.append(
            "Utility function is not defined. LCM expects a function called 'utility'"
            "in the functions dictionary.",
        )

    if states_without_next_func := [
        state for state in model.states if f"next_{state}" not in model.functions
    ]:
        error_messages.append(
            "Each state must have a corresponding next state function. For the "
            "following states, no next state function was found: "
            f"{states_without_next_func}.",
        )

    if states_and_choices_overlap := set(model.states) & set(model.choices):
        error_messages.append(
            "States and choices cannot have overlapping names. The following names "
            f"are used in both states and choices: {states_and_choices_overlap}.",
        )

    return error_messages


# Discrete grid
# ======================================================================================


def _validate_discrete_grid(options: Collection[ScalarUserInput]) -> list[str]:
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

    # if list(options) != list(range(len(options))):
    #     error_messages.append(
    #         "options must be a list of consecutive integers starting from 0",
    #     )

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
