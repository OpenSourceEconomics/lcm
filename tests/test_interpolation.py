import itertools

import numpy as np
from lcm.interpolation import linear_interpolation
from numpy.testing import assert_array_almost_equal as aaae
from scipy.interpolate import RegularGridInterpolator


def f(a, b, c):
    return 2 * a ** 3 + 3 * b ** 2 - c


def g(a, b, c, d):
    return f(a, b, c) - d


def h(a, b, c, d, e):
    return g(a, b, c, d) + e ** 5


def test_linear_interpolation_2d():
    grid1 = np.array([1, 2, 3, 4, 5.0])
    grid2 = np.array([2, 3, 4.0])

    prod_grid = np.array(list(itertools.product(grid1, grid2)))
    values = (prod_grid ** 2).sum(axis=1).reshape(5, 3)

    points = np.array([[2.5, 3.5], [2.1, 3.8], [2.7, 3.3]])

    grid_info = [("linspace", (1, 5, 5)), ("linspace", (2, 4, 3))]

    for point in points:
        calculated = linear_interpolation(
            values=values,
            point=point,
            grid_info=grid_info,
        )

        scipy_func = RegularGridInterpolator(
            points=(grid1, grid2), values=values, method="linear"
        )
        scipy_res = scipy_func(point)

        aaae(calculated, scipy_res)


def test_linear_interpolation_3d():
    grid1 = np.linspace(1, 5, 5)
    grid2 = np.linspace(4, 7, 4)
    grid3 = np.linspace(7, 9, 2)

    values = f(*np.meshgrid(grid1, grid2, grid3, indexing="ij", sparse=False))

    points = np.array([[2.1, 6.2, 8.3], [3.3, 5.2, 7.1], [2.7, 4.3, 7]])

    grid_info = [
        ("linspace", (1, 5, 5)),
        ("linspace", (4, 7, 4)),
        ("linspace", (7, 9, 2)),
    ]

    for point in points:
        calculated = linear_interpolation(
            values=values,
            point=point,
            grid_info=grid_info,
        )
        scipy_func = RegularGridInterpolator(
            points=(grid1, grid2, grid3), values=values, method="linear"
        )
        scipy_res = scipy_func(point)

        aaae(calculated, scipy_res)


def test_linear_interpolation_4d():
    grid1 = np.linspace(1, 5, 5)
    grid2 = np.linspace(4, 7, 4)
    grid3 = np.linspace(7, 9, 2)
    grid4 = np.linspace(10, 11, 2)

    values = g(*np.meshgrid(grid1, grid2, grid3, grid4, indexing="ij", sparse=False))

    points = np.array([[2.1, 6.2, 8.3, 10.4], [3.3, 5.2, 7.1, 10], [2.7, 4.3, 7, 11.0]])

    grid_info = [
        ("linspace", (1, 5, 5)),
        ("linspace", (4, 7, 4)),
        ("linspace", (7, 9, 2)),
        ("linspace", (10, 11, 2)),
    ]

    for point in points:
        calculated = linear_interpolation(
            values=values,
            point=point,
            grid_info=grid_info,
        )
        scipy_func = RegularGridInterpolator(
            points=(grid1, grid2, grid3, grid4), values=values, method="linear"
        )
        scipy_res = scipy_func(point)

        aaae(calculated, scipy_res)


def test_linear_interpolation_5d():
    grid1 = np.linspace(1, 5, 5)
    grid2 = np.linspace(4, 7, 4)
    grid3 = np.linspace(7, 9, 2)
    grid4 = np.linspace(10, 11, 2)
    grid5 = np.linspace(-3, 4, 10)

    values = h(
        *np.meshgrid(grid1, grid2, grid3, grid4, grid5, indexing="ij", sparse=False)
    )

    points = np.array(
        [[2.1, 6.2, 8.3, 10.4, -3], [3.3, 5.2, 7.1, 10, 0], [2.7, 4.3, 7, 11.0, -1.7]]
    )

    grid_info = [
        ("linspace", (1, 5, 5)),
        ("linspace", (4, 7, 4)),
        ("linspace", (7, 9, 2)),
        ("linspace", (10, 11, 2)),
        ("linspace", (-3, 4, 10)),
    ]

    for point in points:
        calculated = linear_interpolation(
            values=values,
            point=point,
            grid_info=grid_info,
        )
        scipy_func = RegularGridInterpolator(
            points=(grid1, grid2, grid3, grid4, grid5), values=values, method="linear"
        )
        scipy_res = scipy_func(point)

        aaae(calculated, scipy_res)
