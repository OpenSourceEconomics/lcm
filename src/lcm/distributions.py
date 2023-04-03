"""Collection of distributions for shocks."""
import numpy as np


def lognormal(sd, size):
    return np.random.lognormal(0, sd, size)


def get_lognormal_params():
    return {"sd": np.nan}


def extreme_value(scale, size):
    return np.random.gumbel(0, scale, size)


def get_extreme_value_params():
    return {"scale": np.nan}
