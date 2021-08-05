import numpy as np


def linspace(start, stop, n_points):
    return np.linspace(start, stop, n_points)


def linspace_value_to_index(start, stop, n_points, value):  # noqa: U100
    raise NotImplementedError()


def linspace_index_to_value(start, stop, n_points, index):  # noqa: U100
    raise NotImplementedError()
