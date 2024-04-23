import numpy as np
from lcm.grids import (
    get_linspace_coordinate,
    get_logspace_coordinate,
    linspace,
    logspace,
)
from numpy.testing import assert_array_almost_equal as aaae


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
