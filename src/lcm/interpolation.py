import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

import lcm.grids as grids_module


def linear_interpolation(values, point, grid_info):
    """Interpolate values at point using linear interpolation.

    Args:
        values (array): Values to interpolate.
        point (list): List of points to interpolate at. Must have same length as
            grid_info.
        grid_info (list): List with tuple entries (grid_type, (start, stop, n_points)).
            n_points must be the same as length of values.

    Returns:
        Interpolated values.

    """
    mapped_values = []
    for i, (grid_type, args) in enumerate(grid_info):
        func = getattr(grids_module, f"get_{grid_type}_coordinate")
        mapped_values.append(func(point[i], *args))

    mapped_point = jnp.array(mapped_values)
    return map_coordinates(
        input=values,
        coordinates=mapped_point,
        order=1,
        mode="nearest",
    )
