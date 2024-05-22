"""Functions to generate and work with different kinds of grids.

Grid generation functions must have the following signature:

    Signature (start: ScalarNumeric, stop: ScalarNumeric, n_points: int) -> jax.Array

They take start and end points and create a grid of points between them.


Interpolation info functions must have the following signature:

    Signature (
        value: ScalarNumeric,
        start: ScalarNumeric,
        stop: ScalarNumeric,
        n_points: int
    ) -> ScalarNumeric

They take the information required to generate a grid, and return an index corresponding
to the value, which is a point in the space but not necessarily a grid point.

Some of the arguments will not be used by all functions but the aligned interface makes
it easy to call functions interchangeably.

"""

import jax.numpy as jnp
from jax import Array

# The functions in this module are designed to work with scalar numerical values. This
# also includes zero dimensional jax arrays.
ScalarNumeric = int | float | Array


def linspace(start: ScalarNumeric, stop: ScalarNumeric, n_points: int) -> Array:
    """Wrapper around jnp.linspace.

    Returns a linearly spaced grid between start and stop with n_points, including both
    endpoints.

    """
    return jnp.linspace(start, stop, n_points)


def get_linspace_coordinate(
    value: ScalarNumeric,
    start: ScalarNumeric,
    stop: ScalarNumeric,
    n_points: int,
) -> ScalarNumeric:
    """Map a value into the input needed for jax.scipy.ndimage.map_coordinates."""
    step_length = (stop - start) / (n_points - 1)
    return (value - start) / step_length


def logspace(start: ScalarNumeric, stop: ScalarNumeric, n_points: int) -> Array:
    """Wrapper around jnp.logspace.

    Returns a logarithmically spaced grid between start and stop with n_points,
    including both endpoints.

    """
    start_lin = jnp.log(start)
    stop_lin = jnp.log(stop)
    return jnp.logspace(start_lin, stop_lin, n_points, base=jnp.e)


def get_logspace_coordinate(
    value: ScalarNumeric,
    start: ScalarNumeric,
    stop: ScalarNumeric,
    n_points: int,
) -> ScalarNumeric:
    """Map a value into the input needed for jax.scipy.ndimage.map_coordinates."""
    start_lin = jnp.log(start)
    stop_lin = jnp.log(stop)
    value_lin = jnp.log(value)

    mapped_point_lin = get_linspace_coordinate(value_lin, start_lin, stop_lin, n_points)

    # Calculate lower and upper point on log/exp scale
    step_length = (stop_lin - start_lin) / (n_points - 1)
    rank_lower_gridpoint = jnp.floor(mapped_point_lin)
    rank_upper_gridpoint = rank_lower_gridpoint + 1

    # Calc
    lower_gridpoint = jnp.exp(start_lin + step_length * rank_lower_gridpoint)
    upper_gridpoint = jnp.exp(start_lin + step_length * rank_upper_gridpoint)

    # Calculate transformed mapped point
    decimal = (value - lower_gridpoint) / (upper_gridpoint - lower_gridpoint)
    return rank_lower_gridpoint + decimal
