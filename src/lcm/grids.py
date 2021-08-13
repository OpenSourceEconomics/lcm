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


def logspace(start, stop, n_points):
    start_lin = jnp.log(start)
    stop_lin = jnp.log(stop)
    return jnp.logspace(start_lin, stop_lin, n_points, base=2.718281828459045)


def get_logspace_coordinate(value, start, stop, n_points):
    """Map a value into the input needed for map_coordinates."""
    start_lin = jnp.log(start)
    stop_lin = jnp.log(stop)
    value_lin = jnp.log(value)
    mapped_point_log = get_linspace_coordinate(value_lin, start_lin, stop_lin, n_points)

    # Calculate mapped point on a linear scale
    step_length = (stop_lin - start_lin) / (n_points - 1)
    lower_point = jnp.exp(start_lin + step_length * jnp.floor(mapped_point_log))
    upper_point = lower_point + 1

    mapped_point_lin = lower_point + (value - lower_point) / (upper_point - lower_point)
    return mapped_point_lin
