"""Collection of distributions for shocks."""
import numpy as np


def lognormal(sd, size):
    return np.random.lognormal(0, sd, size)  # noqa: NPY002


def get_lognormal_params():
    return {"sd": np.nan}


def extreme_value(scale, size):
    return np.random.gumbel(0, scale, size)  # noqa: NPY002


def get_extreme_value_params():
    return {"scale": np.nan}
