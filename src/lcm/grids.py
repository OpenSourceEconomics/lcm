"""Collection of classes that are used by the user to define the model and grids."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array

from lcm import grid_helpers
from lcm.exceptions import GridInitializationError, format_messages
from lcm.typing import Scalar


class Grid(ABC):
    """LCM Grid base class."""

    @abstractmethod
    def to_jax(self) -> Array:
        """Convert the grid to a Jax array."""


class DiscreteGrid(Grid):
    """A class representing a discrete grid.

    Args:
        category_class (type): The category class representing the grid categories. Must
            be a dataclass with fields that have unique scalar int or float values.

    Attributes:
        categories: The list of category names.
        codes: The list of category codes.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with scalar
            int or float fields.

    """

    def __init__(self, category_class: type) -> None:
        """Initialize the DiscreteGrid.

        Args:
            category_class (type): The category class representing the grid categories.
                Must be a dataclass with fields that have unique scalar int or float
                values.

        """
        _validate_discrete_grid(category_class)

        names_and_values = _get_field_names_and_values(category_class)

        self.__categories = tuple(names_and_values.keys())
        self.__codes = tuple(names_and_values.values())

    @property
    def categories(self) -> tuple[str, ...]:
        """Get the list of category names."""
        return self.__categories

    @property
    def codes(self) -> tuple[int | float, ...]:
        """Get the list of category codes."""
        return self.__codes

    def to_jax(self) -> Array:
        """Convert the grid to a Jax array."""
        return jnp.array(self.codes)


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


def _validate_discrete_grid(category_class: type) -> None:
    """Validate the field names and values of the category_class passed to DiscreteGrid.

    Args:
        category_class: The category class representing the grid categories. Must
            be a dataclass with fields that have unique scalar int or float values.

    Raises:
        GridInitializationError: If the `category_class` is not a dataclass with scalar
            int or float fields.

    """
    if not is_dataclass(category_class):
        raise GridInitializationError(
            "category_class must be a dataclass with scalar int or float fields, "
            f"but is {category_class}."
        )

    names_and_values = _get_field_names_and_values(category_class)

    error_messages = []

    if not names_and_values:
        error_messages.append(
            "category_class passed to DiscreteGrid must have at least one field"
        )

    names_with_non_numerical_values = [
        name
        for name, value in names_and_values.items()
        if not isinstance(value, int | float)
    ]
    if names_with_non_numerical_values:
        error_messages.append(
            "Field values of the category_class passed to DiscreteGrid can only be "
            "scalar int or float values. The values to the following fields are not: "
            f"{names_with_non_numerical_values}"
        )

    values = list(names_and_values.values())
    duplicated_values = [v for v in values if values.count(v) > 1]
    if duplicated_values:
        error_messages.append(
            "Field values of the category_class passed to DiscreteGrid must be unique. "
            "The following values are duplicated: "
            f"{set(duplicated_values)}"
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)
    if error_messages:
        msg = format_messages(error_messages)
        raise GridInitializationError(msg)


def _get_field_names_and_values(dc: type) -> dict[str, Any]:
    """Get the fields of a dataclass.

    Args:
        dc: The dataclass to get the fields of.

    Returns:
        A dictionary with the field names as keys and the field values as values. If
        no value is provided for a field, the value is set to None.

    """
    return {field.name: getattr(dc, field.name, None) for field in fields(dc)}


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
