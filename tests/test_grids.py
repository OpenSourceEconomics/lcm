from dataclasses import make_dataclass

import numpy as np
import pytest

from lcm.exceptions import GridInitializationError
from lcm.grids import (
    DiscreteGrid,
    LinspaceGrid,
    LogspaceGrid,
    _get_field_names_and_values,
    _validate_continuous_grid,
    _validate_discrete_grid,
)


def test_validate_discrete_grid_empty():
    names_and_values = {}
    expected = ["category_class passed to DiscreteGrid must have at least one field"]
    assert _validate_discrete_grid(names_and_values) == expected


def test_validate_discrete_grid_non_scalar_input():
    names_and_values = {"a": 1, "b": "wrong_type"}
    expected = [
        (
            "Field values of the category_class passed to DiscreteGrid can only be "
            "scalar int or float values. The values to the following fields are not: "
            "['b']"
        )
    ]
    assert _validate_discrete_grid(names_and_values) == expected


def test_validate_discrete_grid_none_input():
    names_and_values = {"a": None, "b": 1}
    expected = [
        (
            "Field values of the category_class passed to DiscreteGrid can only be "
            "scalar int or float values. The values to the following fields are not: "
            "['a']"
        )
    ]
    assert _validate_discrete_grid(names_and_values) == expected


def test_validate_discrete_grid_non_unique():
    names_and_values = {"a": 1, "b": 2, "c": 2}
    expected = [
        (
            "Field values of the category_class passed to DiscreteGrid must be unique. "
            "The following values are duplicated: {2}"
        )
    ]
    assert _validate_discrete_grid(names_and_values) == expected


def test_get_fields_with_defaults():
    category_class = make_dataclass("Category", [("a", int, 1), ("b", int, 2)])
    assert _get_field_names_and_values(category_class) == {"a": 1, "b": 2}


def test_get_fields_no_defaults():
    category_class = make_dataclass("Category", [("a", int), ("b", int)])
    assert _get_field_names_and_values(category_class) == {"a": None, "b": None}


def test_get_fields_instance():
    category_class = make_dataclass("Category", [("a", int), ("b", int)])
    assert _get_field_names_and_values(category_class(a=1, b=2)) == {"a": 1, "b": 2}


def test_get_fields_empty():
    category_class = make_dataclass("Category", [])
    assert _get_field_names_and_values(category_class) == {}


def test_validate_continuous_grid_invalid_start():
    assert _validate_continuous_grid("a", 1, 10) == [
        "start must be a scalar int or float value"
    ]


def test_validate_continuous_grid_invalid_stop():
    assert _validate_continuous_grid(1, "a", 10) == [
        "stop must be a scalar int or float value"
    ]


def test_validate_continuous_grid_invalid_n_points():
    assert _validate_continuous_grid(1, 2, "a") == [
        "n_points must be an int greater than 0 but is a"
    ]


def test_validate_continuous_grid_negative_n_points():
    assert _validate_continuous_grid(1, 2, -1) == [
        "n_points must be an int greater than 0 but is -1"
    ]


def test_validate_continuous_grid_start_greater_than_stop():
    assert _validate_continuous_grid(2, 1, 10) == ["start must be less than stop"]


def test_linspace_grid_creation():
    grid = LinspaceGrid(start=1, stop=5, n_points=5)
    assert np.allclose(grid.to_jax(), np.linspace(1, 5, 5))


def test_logspace_grid_creation():
    grid = LogspaceGrid(start=1, stop=10, n_points=3)
    assert np.allclose(grid.to_jax(), np.logspace(np.log10(1), np.log10(10), 3))


def test_discrete_grid_creation():
    category_class = make_dataclass(
        "Category", [("a", int, 0), ("b", int, 1), ("c", int, 2)]
    )
    grid = DiscreteGrid(category_class)
    assert np.allclose(grid.to_jax(), np.arange(3))


def test_linspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LinspaceGrid(start=1, stop=0, n_points=10)


def test_logspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LogspaceGrid(start=1, stop=0, n_points=10)


def test_discrete_grid_invalid_category_class():
    category_class = make_dataclass(
        "Category", [("a", int, 1), ("b", str, "wrong_type")]
    )
    with pytest.raises(
        GridInitializationError,
        match="Field values of the category_class passed to DiscreteGrid can only be",
    ):
        DiscreteGrid(category_class)
