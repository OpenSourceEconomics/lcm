from functools import partial

import jax.numpy as jnp
import lcm.grids as grids_module
import numpy as np
from dags import concatenate_functions
from dags.signature import with_signature
from jax.scipy.ndimage import map_coordinates


def get_function_evaluator(
    space_info,
    data_name,
    interpolation_options=None,
    return_type="function",
    input_prefix="",
    out_name="__fval__",
):
    """Create a function to look-up and interpolate a function pre-calculated on a grid.

    An example of a pre-calculated function is a value or policy function. These are
    evaluated on the space of all sparse and dense discrete state variables as well as
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

    - Translate values of discrete variables into positions
    - Look up the position of sparse variables in an indexer array.
    - Index into the array with the precalculated function values to extract only the
      part on which interpolation is needed.
    - Tranlate values of continuous variables into coordinates needed for interpolation
      via scipy.ndimage.map_coordinates.
    - Do the actual interpolation.

    Depending on the grid, only a subset of these steps is relevant. The chosen
    implementation of each step is also adjusted to the type of grid. In particular
    we try to avoid searching for neighboring values on a grid and instead exploit
    structure in the grid to calculate where those entries are. The order
    in which the functions are called is determined by a DAG.

    Args:
        space_info (SpaceInfo): Namedtuple containing all information needed to
            interpret the precalculated values of a function.
        data_name (str): The name of the argument via which the precalculated values
            will be passed into the resulting function.
        interpolation_options (dict): Dictionary of keyword arguments for
            interpolation via map_coordinates.
        return_type (str): Either "function" or "dict". If return_type="function",
            return a function. If return_type="dict", return a dictionary with all
            inputs needed to call `dags.concatenate_functions`.
        input_prefix (str): Prefix that will be added to all argument names of the
            resulting function, except for the helpers arguments such as indexers
            or value arrays. Default is the empty string. The prefix needs to contain
            the separator. E.g. `next_` if an undescore should be used as separator.

    Returns:
        callable: A callable that lets you evaluate a function defined be precalculated
            values on space formed by discrete and continuous grids.

    """
    funcs = {}

    # ==================================================================================
    # check inputs
    # ==================================================================================
    _fail_if_interpolation_axes_are_not_last(space_info)
    _need_interpolation = bool(space_info.interpolation_info)

    # ==================================================================================
    # create functions to look up position of discrete variables from labels
    # ==================================================================================
    for var, labels in space_info.lookup_info.items():
        funcs[f"__{var}_pos__"] = _get_label_translator(
            labels=labels, in_name=input_prefix + var
        )

    # ==================================================================================
    # wrap the indexers and put them it into funcs
    # ==================================================================================
    for indexer in space_info.indexer_infos:
        funcs[f"__{indexer.out_name}_pos__"] = _get_lookup_function(
            array_name=indexer.name,
            axis_names=[f"__{var}_pos__" for var in indexer.axis_names],
        )

    # ==================================================================================
    # create a function for the discrete lookup
    # ==================================================================================
    # lookup is positional, so the inputs of the wrapper functions need to be the
    # outcomes of tranlating labels into positions
    _internal_axes = [f"__{var}_pos__" for var in space_info.axis_names]
    _lookup_axes = [var for var in _internal_axes if var in funcs]

    _out_name = "__interpolation_data__" if _need_interpolation else "__fval__"
    funcs[_out_name] = _get_lookup_function(
        array_name=data_name,
        axis_names=_lookup_axes,
    )

    if _need_interpolation:
        # ==============================================================================
        # create functions to find coordinates for the interpolation
        # ==============================================================================
        for var, grid_info in space_info.interpolation_info.items():
            funcs[f"__{var}_coord__"] = _get_coordinate_finder(
                in_name=input_prefix + var,
                grid_type=grid_info.kind,
                grid_info=grid_info.specs,
            )

        # ==============================================================================
        # create interpolation function
        # ==============================================================================
        _interpolation_axes = [
            f"__{var}_coord__"
            for var in space_info.axis_names
            if var in space_info.interpolation_info
        ]
        funcs[out_name] = _get_interpolator(
            data_name="__interpolation_data__",
            axis_names=_interpolation_axes,
            map_coordinates_kwargs=interpolation_options,
        )

    # ==================================================================================
    # prepare the output
    # ==================================================================================
    if return_type == "function":
        out = concatenate_functions(
            functions=funcs,
            targets=out_name,
        )
    elif return_type == "dict":
        out = {"functions": funcs, "targets": out_name}
    else:
        raise ValueError(
            f"return_type must be either 'function' or 'dict', but got {return_type}."
        )

    return out


def _get_label_translator(labels, in_name):
    """Create a function that translates a label into a position in a list of labels.

    Args:
        labels (list, np.ndarray): List of allowed labels.
        in_name (str): Name of the variable that provides the label in the signature
            of the resulting function.

    Returns:
        callable: A callable with the keyword only argument ``[in_name]`` that returns
            converts a label into a position in a list of labels.

    """
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

    return translate_label


def _get_lookup_function(array_name, axis_names):
    """Create a function that emulates indexing into an array via named axes.

    Args:
        array_name (str): The name of the array into which the function indexes.
        axis_names (list): List of strings with names for each axis in the array.

    Returns:
        callable: A callable with the keyword-only arguments [axis_names] + [name]
            that looks up values in an indexer array called ``name``.

    """

    @with_signature(kwargs=axis_names + [array_name])
    def lookup_wrapper(**kwargs):
        positions = tuple(kwargs[var] for var in axis_names)
        arr = kwargs[array_name]
        return arr[positions]

    return lookup_wrapper


def _get_coordinate_finder(in_name, grid_type, grid_info):
    """Create a function that translates a value into coordinates on a grid.

    The resulting coordinates can be used to do linear interpolation via
    scipy.ndimage.map_coordinates.

    Args:
        in_name (str): Name via which the value to be translated into coordinates
            will be passed into the resulting function.
        grid_type (str): Type of the grid, e.g. "linspace" or "logspace". The type of
            grid must be implemented in lcm.grids.
        grid_info (dict): Dict with information that defines the grid. E.g. for a
            linspace those are {"start": float, "stop": float, "n_points": int}. See
            lcm.grids for details.

    Returns:
        callable: A callable with keyword-only argument [in_name] that translates a
            value into coordinates on a grid.

    """
    grid_info = {} if grid_info is None else grid_info

    raw_func = getattr(grids_module, f"get_{grid_type}_coordinate")
    partialled_func = partial(raw_func, **grid_info)

    @with_signature(kwargs=[in_name])
    def find_coordinate(**kwargs):
        return partialled_func(kwargs[in_name])

    return find_coordinate


def _get_interpolator(data_name, axis_names, map_coordinates_kwargs=None):
    """Create a function interpolator via named axes.

    Args:
        data_name (str): Name of the argument via which function values on which the
            interpolation is done are passed into the interpolator.
        axis_names (str): Names of the axes in the data array.
        map_coordinates_kwargs (dict): Keyword arguments for
            scipy.ndimage.map_coordinates.

    Returns:
        callable: A callable that interpolates a function via named axes.

    """
    kwargs = {"order": 1, "mode": "nearest"}
    if map_coordinates_kwargs is not None:
        kwargs = {**kwargs, **map_coordinates_kwargs}

    partialled_map_coordinates = partial(map_coordinates, **kwargs)

    @with_signature(kwargs=[data_name] + axis_names)
    def interpolate(**kwargs):
        coordinates = jnp.array([kwargs[var] for var in axis_names])
        out = partialled_map_coordinates(
            input=kwargs[data_name],
            coordinates=coordinates,
        )
        return out

    return interpolate


def _fail_if_interpolation_axes_are_not_last(space_info):
    """Fail if the interpolation axes are not the last elements in axis_names."""
    common = set(space_info.interpolation_info) & set(space_info.axis_names)

    if common:
        n_common = len(common)
        if sorted(common) != sorted(space_info.axis_names[-n_common:]):
            raise ValueError(
                "Interpolation axes need to be the last entries in axis_order."
            )
