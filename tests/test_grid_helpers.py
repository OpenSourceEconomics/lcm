import numpy as np
import pytest
from jax.scipy.ndimage import map_coordinates
from numpy.testing import assert_array_almost_equal as aaae

from lcm.grid_helpers import (
    get_linspace_coordinate,
    get_logspace_coordinate,
    linspace,
    logspace,
)


def test_linspace():
    calculated = linspace(start=1, stop=2, n_points=6)
    expected = np.array([1, 1.2, 1.4, 1.6, 1.8, 2])
    aaae(calculated, expected)


def test_linspace_mapped_value():
    """For reference of the grid values, see expected grid in `test_linspace`."""
    # Get position corresponding to a value in the grid
    calculated = get_linspace_coordinate(
        value=1.2,
        start=1,
        stop=2,
        n_points=6,
    )
    assert np.allclose(calculated, 1.0)

    # Get position corresponding to a value that is between two grid points
    # ----------------------------------------------------------------------------------
    # Here, the value is 1.3, that is in the middle of 1.2 and 1.4, which have the
    # positions 1 and 2, respectively. Therefore, we want the position to be 1.5.
    calculated = get_linspace_coordinate(
        value=1.3,
        start=1,
        stop=2,
        n_points=6,
    )
    assert np.allclose(calculated, 1.5)

    # Get position corresponding to a value that is outside the grid
    calculated = get_linspace_coordinate(
        value=0.6,
        start=1,
        stop=2,
        n_points=6,
    )
    assert np.allclose(calculated, -2.0)


def test_logspace():
    calculated = logspace(start=1, stop=100, n_points=7)
    expected = np.array(
        [1.0, 2.15443469, 4.64158883, 10.0, 21.5443469, 46.41588834, 100.0],
    )
    aaae(calculated, expected)


def test_logspace_mapped_value():
    """For reference of the grid values, see expected grid in `test_logspace`."""
    calculated = get_logspace_coordinate(
        value=(2.15443469 + 4.64158883) / 2,
        start=1,
        stop=100,
        n_points=7,
    )
    assert np.allclose(calculated, 1.5)


@pytest.mark.illustrative
def test_map_coordinates_linear():
    """Illustrative test on how the output of get_linspace_coordinate can be used."""
    grid_info = {
        "start": 0,
        "stop": 1,
        "n_points": 3,
    }

    grid = linspace(**grid_info)  # [0, 0.5, 1]

    values = 2 * grid  # [0, 1.0, 2.0]

    # We choose a coordinate that is exactly in the middle between the first and second
    # entry of the grid.
    coordinate = get_linspace_coordinate(
        value=0.25,
        **grid_info,
    )

    # Perform the linear interpolation
    interpolated_value = map_coordinates(values, [coordinate], order=1, mode="nearest")
    assert np.allclose(interpolated_value, 0.5)


@pytest.mark.illustrative
def test_map_coordinates_logarithmic():
    """Illustrative test on how the output of get_logspace_coordinate can be used."""
    grid_info = {
        "start": 1,
        "stop": 2,
        "n_points": 3,
    }

    grid = logspace(**grid_info)  # [1.0, 1.414213562373095, 2.0]

    values = 2 * grid  # [2.0, 2.82842712474619, 4.0]

    # We choose a coordinate that is exactly in the middle between the first and second
    # entry of the grid.
    coordinate = get_logspace_coordinate(
        value=(1.0 + 1.414213562373095) / 2,
        **grid_info,
    )

    # Perform the linear interpolation
    interpolated_value = map_coordinates(values, [coordinate], order=1, mode="nearest")
    assert np.allclose(interpolated_value, (2.0 + 2.82842712474619) / 2)


@pytest.mark.illustrative
def test_map_coordinates_linear_outside_grid():
    """Illustrative test on what happens to values outside the grid.

    If mode="nearest", the value corresponding to the closest coordinate that still lies
    within the grid is returned.

    """
    grid_info = {
        "start": 0,
        "stop": 1,
        "n_points": 2,
    }

    grid = linspace(**grid_info)  # [0, 1]

    values = 2 * grid  # [0, 2.0]

    # We choose a coordinate that is exactly in the middle between the first and second
    # entry of the grid.
    coordinate = get_linspace_coordinate(
        value=-1,
        **grid_info,
    )

    assert coordinate == -1.0

    # Perform the linear interpolation
    interpolated_value = map_coordinates(values, [coordinate], order=1, mode="nearest")

    # Because mode="nearest", the value at the first grid point is returned
    assert np.allclose(interpolated_value, 0.0)
