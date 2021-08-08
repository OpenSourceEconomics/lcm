import itertools

import numpy as np
from lcm.interpolation import linear_interpolation
from scipy.interpolate import interp2d


def test_linear_interpolation_2d():
    grid1 = np.array([1, 2, 3, 4, 5.0])
    grid2 = np.array([2, 3, 4.0])

    prod_grid = np.array(list(itertools.product(grid1, grid2)))
    values = (prod_grid ** 2).sum(axis=1).reshape(5, 3)

    points = np.array([[2.5, 3.5], [2.1, 3.8], [2.7, 3.3]])

    grid_info = [("linspace", (1, 5, 5)), ("linspace", (2, 4, 3))]

    for point in points:
        point = np.array([2.5, 3.5])

        calculated = linear_interpolation(
            values=values.T,
            point=point,
            grid_info=grid_info,
        )

        scipy_func = interp2d(
            x=grid2,
            y=grid1,
            z=values,
        )
        scipy_res = scipy_func(*point)

        assert np.allclose(calculated, scipy_res)
