import numpy as np
from lcm.grids import get_linspace_coordinate
from lcm.grids import linspace
from numpy.testing import assert_array_almost_equal as aaae


def test_linspace():
    calculated = linspace(start=1, stop=2, n_points=6)
    expected = np.array([1, 1.2, 1.4, 1.6, 1.8, 2])
    aaae(calculated, expected)


def test_linspace_mapped_value():
    calculated = get_linspace_coordinate(
        value=1.3,
        start=1,
        stop=2,
        n_points=6,
    )

    expected = 1.5

    assert np.allclose(calculated, expected)
