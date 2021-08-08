"""Functions to generate and work with different kinds of grids.

Grid generation functions have the arguments:

- start
- stop
- n_points

interpolation info functions have the arguments

- value
- start
- stop
- n_points
- grid

Some of the arguments will not be used by all functions but the aligned interface makes
it easy to call functions interchangeably.

"""
import jax.numpy as jnp


def linspace(start, stop, n_points):
    return jnp.linspace(start, stop, n_points)


def get_linspace_coordinate(value, start, stop, n_points):
    """Map a value into the input needed for map_coordinates."""
    step_length = (stop - start) / (n_points - 1)
    mapped_point = (value - start) / step_length
    return mapped_point
