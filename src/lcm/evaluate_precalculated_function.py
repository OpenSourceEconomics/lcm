from functools import partial

import jax.numpy as jnp
import lcm.grids as grids_module
import numpy as np
from dags import concatenate_functions
from dags.signature import with_signature
from jax.scipy.ndimage import map_coordinates


def get_precalculated_function_evaluator(
    discrete_info,
    continuous_info,
    indexer_info,
    axis_order,
    data_name,
    interpolation_options=None,
):
    """Create a function to look-up and interpolate a function pre-calculated on a grid.

    An example of a pre-calculated function is a value or policy function. These are
    evaluated on the grid of all sparse and dense discrete state variables as well as
    all continuous state variables.

    This function dynamically generates a function that looks up and interpolates values
    of the pre-calculated function. The arguments of the resulting function can be
    split in two categories:
    1. Helper arguments such as information about the grid, indexer arrays and the
    pre-calculated values of the function.
    2. The original arguments of the function that was pre-calculated on the grid.

    After partialling in all helper arguments, the resulting function will completely
    hide that it is not analytical and feel like a normal function. In particular, it
    can be jitted, differentiated and vmapped with jax.

    The resulting function roughly does the following steps:

    - Translate values of discrete variables into positions in a cartesian product
      of all discrete variables.
    - Translate the positions of sparse variables to the position in our internal sparse
      array representation;
    - Index into the ``values_`` array to extract only the part that is relevant for the
      given values of discrete variables.
    - Index into the ``grids_`` array to extract the grid that is relevant for the given
      values of discrete variables.
    - Translate the values of the continuous variables coordinates needed for
      map_coordinates
    - Do the actual interpolation.

    Depending on the grid, only a subset of these steps is relevant. The chosen
    implementation of each step is also adjusted to the type of grid. The order
    in which the functions are called is determined by a DAG.



    Args:


    Returns:
        function:


    """

    functions = {}

    # create functions to look up position of discrete variables from labels
    for var, labels in discrete_info.items():
        _out_name = f"__{var}_pos__"
        functions[_out_name] = get_label_translator(
            labels=labels, in_name=var, out_name=_out_name
        )

    # wrap the indexer and put it into functions
    _out_name = f"__{indexer_info.out_name}_pos__"
    functions[_out_name] = get_lookup_function(
        array_name=indexer_info.name,
        axis_order=[f"__{var}_pos__" for var in indexer_info.axis_order],
        out_name=_out_name,
    )

    # create a function for the discrete lookup
    _internal_axes = [f"__{var}_pos__" for var in axis_order]
    _lookup_axes = [var for var in _internal_axes if var in functions]

    _out_name = "__interpolation_data__"
    functions[_out_name] = get_lookup_function(
        array_name=data_name,
        axis_order=_lookup_axes,
        out_name=_out_name,
    )

    # create functions to find coordinates for the interpolation
    for var, grid_info in continuous_info.items():
        _out_name = f"__{var}_coord__"
        functions[_out_name] = get_coordinate_finder(
            in_name=var,
            grid_type=grid_info.kind,
            grid_info=grid_info.specs,
            out_name=_out_name,
        )

    # create interpolation function
    functions["__fval__"] = get_interpolator(
        value_name="__interpolation_data__",
        axis_order=[f"__{var}_coord__" for var in continuous_info],
        map_coordinates_kwargs=interpolation_options,
    )

    # build the dag
    evaluator = concatenate_functions(
        functions=functions,
        targets="__fval__",
    )

    return evaluator


def get_lookup_function(array_name, axis_order, out_name=None):
    """Create a function tha enables name based lookups in an array.

    Args:
        indexer_name (str): The name of the indexer. This will be the corresponding
            argument name in the wrapper function.
        axis_order (list): List of strings with names for each axis in the indexer.


    Returns:
        callable: A callable with the keyword-only arguments [axis_order] + [name]
            that looks up values in an indexer array called ``name``.

    """

    if out_name is None:
        out_name = f"{array_name.replace('_indexer', '')}_pos"

    @with_signature(kwargs=axis_order + [array_name])
    def indexer_wrapper(**kwargs):
        positions = tuple(kwargs[var] for var in axis_order)
        arr = kwargs[array_name]
        return arr[positions]

    indexer_wrapper.__name__ = out_name

    return indexer_wrapper


def get_label_translator(labels, in_name, out_name=None):
    """Create a function

    Args:
        grid (jnp.ndarray): 1d jax array with grid values.
        in_name (str): Name of the variable the grid is representing.

    """
    if out_name is None:
        out_name = f"{in_name}_pos"

    if isinstance(labels, (np.ndarray, jnp.ndarray)):
        # tolist converts jax or numpy specific dtypes to python types
        _grid = labels.tolist()
    elif not isinstance(labels, list):
        _grid = list(labels)
    else:
        _grid = labels

    if _grid == list(range(len(labels))):

        @with_signature(kwargs=[in_name])
        def translate_label(**kwargs):
            return kwargs[in_name]

    else:
        val_to_pos = dict(zip(_grid, range(len(labels))))

        @with_signature(kwargs=[in_name])
        def translate_label(**kwargs):
            return val_to_pos[kwargs[in_name]]

    translate_label.__name__ = out_name

    return translate_label


def get_coordinate_finder(in_name, grid_type, grid_info=None, out_name=None):

    if out_name is None:
        out_name = f"{in_name}_coordinate"

    grid_info = {} if grid_info is None else grid_info

    raw_func = getattr(grids_module, f"get_{grid_type}_coordinate")
    partialled_func = partial(raw_func, **grid_info)

    @with_signature(kwargs=[in_name])
    def find_coordinate(**kwargs):
        return partialled_func(kwargs[in_name])

    find_coordinate.__name__ = out_name

    return find_coordinate


def get_interpolator(value_name, axis_order, map_coordinates_kwargs=None):

    kwargs = {"order": 1, "mode": "nearest"}
    if map_coordinates_kwargs is not None:
        kwargs = {**kwargs, **map_coordinates_kwargs}

    partialled_map_coordinates = partial(map_coordinates, **kwargs)

    @with_signature(kwargs=[value_name] + axis_order)
    def interpolate(**kwargs):
        coordinates = jnp.array([kwargs[var] for var in axis_order])
        out = partialled_map_coordinates(
            input=kwargs[value_name],
            coordinates=coordinates,
        )
        return out

    return interpolate
