"""Collection of distributions for shocks.

WARNING: This is currently only a placeholder before we implement the distributions
in JAX.

"""
import numpy as np


def lognormal(sd, size):
    return np.random.lognormal(mean=0, sigma=sd, size=size)  # noqa: NPY002


def get_lognormal_params():
    return {"sd": np.nan}


def extreme_value(scale, size):
    return np.random.gumbel(loc=0, scale=scale, size=size)  # noqa: NPY002


def get_extreme_value_params():
    return {"scale": np.nan}
