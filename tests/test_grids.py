import numpy as np
import pytest

from lcm.exceptions import GridInitializationError
from lcm.grids import (
    DiscreteGrid,
    LinspaceGrid,
    LogspaceGrid,
    _validate_continuous_grid,
    _validate_discrete_grid,
)


def test_validate_discrete_grid_empty():
    assert _validate_discrete_grid([]) == ["options must contain at least one element"]


def test_validate_discrete_grid_non_scalar_input():
    assert _validate_discrete_grid([1, "a"]) == [
        "options must contain only scalar int or float values",
    ]


def test_validate_discrete_grid_non_unique():
    assert _validate_discrete_grid([1, 2, 2]) == [
        "options must contain unique values",
    ]


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
    grid = DiscreteGrid(options=[0, 1, 2])
    assert np.allclose(grid.to_jax(), np.arange(3))


def test_linspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LinspaceGrid(start=1, stop=0, n_points=10)


def test_logspace_grid_invalid_start():
    with pytest.raises(GridInitializationError, match="start must be less than stop"):
        LogspaceGrid(start=1, stop=0, n_points=10)


def test_discrete_grid_invalid_options():
    with pytest.raises(
        GridInitializationError,
        match="options must contain only scalar int or float values",
    ):
        DiscreteGrid(options=[1, "a"])
