from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array
from jax.scipy.ndimage import map_coordinates

import lcm.grids as grids_module
from lcm.functools import all_as_kwargs
from lcm.interfaces import ContinuousGridInfo, ContinuousGridType, SpaceInfo
from lcm.typing import MapCoordinatesOptions


def get_function_evaluator(
    space_info: SpaceInfo,
    data_name: str,
    *,
    interpolation_options: MapCoordinatesOptions | None = None,
    input_prefix: str = "",
) -> Callable[..., Array]:
    """Create a function to look-up and interpolate a function pre-calculated on a grid.

    An example of a pre-calculated function is a value or policy function. These are
    evaluated on the space of all sparse and dense discrete state variables as well as
    all continuous state variables.

    This function dynamically generates a function that looks up and interpolates values
    of the pre-calculated function. The arguments of the resulting function can be split
    in two categories:
       1. Helper arguments such as information about the grid, indexer arrays and the
          pre-calculated values of the function.
       2. The original arguments of the function that was pre-calculated on the grid.

    After partialling in all helper arguments, the resulting function hides that it is
    not an analytical function. In particular, it can be jitted, differentiated and
    vmapped with jax.

    The resulting function roughly does the following steps:

    - Translate values of discrete variables into positions
    - Look up the position of sparse variables in an indexer array.
    - Index into the array with the pre-calculated function values to extract only the
      part on which interpolation is needed.
    - Translate values of continuous variables into coordinates needed for interpolation
      via jax.scipy.ndimage.map_coordinates.
    - Do the actual interpolation.

    Depending on the grid, only a subset of these steps is relevant. The chosen
    implementation of each step is also adjusted to the type of grid. In particular we
    try to avoid searching for neighboring values on a grid and instead exploit
    structure in the grid to calculate where those entries are. The order in which the
    functions are called is determined by a DAG.

    Args:
        space_info: Namedtuple containing all information needed to interpret the
            pre-calculated values of a function.
        data_name: The name of the argument via which the pre-calculated values
            will be passed into the resulting function. In the value function case,
            this could be 'vf_arr', in which case, one would partial in 'vf_arr' into
            the evaluator.
        interpolation_options: Dictionary of interpolation options that will be passed
            to jax.scipy.ndimage.map_coordinates. If None, DefaultMapCoordinatesOptions
            will be used.
        input_prefix: Prefix that will be added to all argument names of the resulting
            function, except for the helpers arguments such as indexers or value arrays.
            Default is the empty string. The prefix needs to contain the separator. E.g.
            `next_` if an undescore should be used as separator.

    Returns:
        callable: A callable that lets you evaluate a function defined be precalculated
            values on space formed by discrete and continuous grids.

    """
    # ==================================================================================
    # check inputs
    # ==================================================================================
    _fail_if_interpolation_axes_are_not_last(space_info)
    _need_interpolation = bool(space_info.interpolation_info)

    # ==================================================================================
    # create functions to look up position of discrete variables from labels
    # ==================================================================================
    funcs = {}

    for var in space_info.lookup_info:
        funcs[f"__{var}_pos__"] = _get_label_translator(
            in_name=input_prefix + var,
        )

    # ==================================================================================
    # wrap the indexers and put them it into funcs
    # ==================================================================================
    for indexer_info in space_info.indexer_infos:
        funcs[f"__{indexer_info.out_name}_pos__"] = _get_lookup_function(
            array_name=indexer_info.name,
            axis_names=[f"__{var}_pos__" for var in indexer_info.axis_names],
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
        for var, grid_spec in space_info.interpolation_info.items():
            funcs[f"__{var}_coord__"] = _get_coordinate_finder(
                in_name=input_prefix + var,
                grid_type=grid_spec.kind,
                grid_info=grid_spec.info,
            )

        # ==============================================================================
        # create interpolation function
        # ==============================================================================
        _interpolation_axes = [
            f"__{var}_coord__"
            for var in space_info.axis_names
            if var in space_info.interpolation_info
        ]
        funcs["__fval__"] = _get_interpolator(
            data_name="__interpolation_data__",
            axis_names=_interpolation_axes,
            map_coordinates_options=interpolation_options,
        )

    return concatenate_functions(
        functions=funcs,
        targets="__fval__",
    )


def _get_label_translator(
    in_name: str,
) -> Callable[..., Array]:
    """Create a function that translates a label into a position in a list of labels.

    Currently, only labels are supported that are themselves indices. The label
    translator in this case is thus just the identity function.

    Args:
        in_name: Name of the variable that provides the label in the signature of trche
            resulting function.

    Returns:
        callable: A callable with the keyword only argument ``in_name`` that converts a
            label into a position in a list of labels.

    """

    @with_signature(args=[in_name])
    def translate_label(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=[in_name])
        return kwargs[in_name]

    return translate_label


def _get_lookup_function(
    array_name: str,
    axis_names: list[str],
) -> Callable[..., Array]:
    """Create a function that emulates indexing into an array via named axes.

    Args:
        array_name (str): The name of the array into which the function indexes.
        axis_names (list): List of strings with names for each axis in the array.

    Returns:
        callable: A callable with the keyword-only arguments [axis_names] + [array_name]
            that looks up values in an indexer array called ``array_name``.

    """
    arg_names = [*axis_names, array_name]

    @with_signature(args=arg_names)
    def lookup_wrapper(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)
        positions = tuple(kwargs[var] for var in axis_names)
        arr = kwargs[array_name]
        return arr[positions]

    return lookup_wrapper


def _get_coordinate_finder(
    in_name: str,
    grid_type: ContinuousGridType,
    grid_info: ContinuousGridInfo,
) -> Callable[..., Array]:
    """Create a function that translates a value into coordinates on a grid.

    The resulting coordinates can be used to do linear interpolation via
    jax.scipy.ndimage.map_coordinates.

    Args:
        in_name: Name via which the value to be translated into coordinates will be
            passed into the resulting function.
        grid_type: Type of the grid, e.g. "linspace" or "logspace". The type of grid
            must be implemented in lcm.grids.
        grid_info: Information on how to build the grid, e.g. start, stop, and n_points.

    Returns:
        callable: A callable with keyword-only argument [in_name] that translates a
            value into coordinates on a grid.

    """
    raw_func = getattr(grids_module, f"get_{grid_type}_coordinate")
    partialled_func = partial(raw_func, **grid_info._asdict())

    @with_signature(args=[in_name])
    def find_coordinate(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=[in_name])
        return partialled_func(kwargs[in_name])

    return find_coordinate


DefaultMapCoordinatesOptions: MapCoordinatesOptions = {
    "order": 1,
    "mode": "nearest",
    "cval": 0.0,
}


def _get_interpolator(
    data_name: str,
    axis_names: list[str],
    map_coordinates_options: MapCoordinatesOptions | None,
) -> Callable[..., Array]:
    """Create a function interpolator via named axes.

    Args:
        data_name: Name of the argument via which function values on which the
            interpolation is done are passed into the interpolator.
        axis_names: Names of the axes in the data array.
        map_coordinates_options: Dictionary of interpolation options that will be passed
            to jax.scipy.ndimage.map_coordinates. If None, DefaultMapCoordinatesOptions
            will be used.

    Returns:
        callable: A callable that interpolates a function via named axes.

    """
    kwargs = DefaultMapCoordinatesOptions | (map_coordinates_options or {})
    partialled_map_coordinates = partial(map_coordinates, **kwargs)

    arg_names = [data_name, *axis_names]

    @with_signature(args=arg_names)
    def interpolate(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)
        coordinates = jnp.array([kwargs[var] for var in axis_names])
        return partialled_map_coordinates(
            input=kwargs[data_name],
            coordinates=coordinates,
        )

    return interpolate


def _fail_if_interpolation_axes_are_not_last(space_info: SpaceInfo) -> None:
    """Fail if the interpolation axes are not the last elements in axis_names.

    Args:
        space_info: Namedtuple containing all information needed to interpret the
            precalculated values of a function.

    Raises:
        ValueError: If the interpolation axes are not the last elements in axis_names.

    """
    common = set(space_info.interpolation_info) & set(space_info.axis_names)

    if common:
        n_common = len(common)
        if sorted(common) != sorted(space_info.axis_names[-n_common:]):
            raise ValueError(
                "Interpolation axes need to be the last entries in axis_order.",
            )
