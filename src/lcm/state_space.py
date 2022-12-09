"""Create a state space for a given model."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
from dags import concatenate_functions
from lcm.dispatchers import productmap
from lcm.dispatchers import spacemap
from lcm.interfaces import IndexerInfo
from lcm.interfaces import Space
from lcm.interfaces import SpaceInfo


def create_state_choice_space(model, period, jit_filter=False):
    """Create a state choice space for the model.

    A state_choice_space is a compressed representation of all feasible states and the
    feasible discrete choices within that state. We currently use the following
    compressions:

    We distinguish between dense and sparse variables (dense_vars and sparse_vars).
    Dense state or choice variables are those whose set of feasible values does not
    depend on any other state or choice variables. Sparse state or choice variables are
    all other state variables. For dense state variables it is thus enough to store the
    grid of feasible values (value_grid), whereas for sparse variables all feasible
    combinations (combination_grid) have to be stored.

    Note:
    -----

    - We only use the filter mask, not the forward mask (yet).

    Args:
        model (Model): A processed model.
        period (int): The period for which the state space is created.
        jit_filter (bool): If True, the filter function is compiled with JAX.

    Returns:
        Space: Space object containing the sparse and dense variables. This can be used
            to execute a function on an entire space.
        SpaceInfo: A SpaceInfo object that contains all information needed to work with
            the output of a function evaluated on the space.
        dict: Dictionary containing state indexer arrays.
        jnp.ndarray: Jax array containing the choice segments needed for the emax
            calculations.

    """
    # ==================================================================================
    # preparations
    # ==================================================================================
    vi = model.variable_info
    has_sparse_states = len(vi.query("is_sparse & is_state")) > 0
    has_sparse_vars = len(vi.query("is_sparse")) > 0
    # ==================================================================================
    # create state choice space
    # ==================================================================================
    _value_grid = _create_value_grid(
        grids=model.grids,
        subset=vi.query("is_dense").index.tolist(),
    )
    if has_sparse_vars:
        _filter_mask = create_filter_mask(
            model=model,
            subset=vi.query("is_sparse").index.tolist(),
            fixed_inputs={"period": period},
            jit_filter=jit_filter,
        )

        _combination_grid = create_combination_grid(
            grids=model.grids,
            masks=_filter_mask,
            subset=vi.query("is_sparse").index.tolist(),
        )
    else:
        _combination_grid = {}

    state_choice_space = Space(
        sparse_vars=_combination_grid,
        dense_vars=_value_grid,
    )
    # ==================================================================================
    # create indexers and segments
    # ==================================================================================
    if has_sparse_vars:
        _state_indexer, _, choice_segments = create_indexers_and_segments(
            mask=_filter_mask,
            n_sparse_states=len(vi.query("is_sparse & is_state")),
        )
    else:
        _state_indexer = None
        choice_segments = None

    if has_sparse_states:
        state_indexers = {}
    else:
        state_indexers = {"state_indexer": _state_indexer}

    # ==================================================================================
    # create state space info
    # ==================================================================================
    # axis_names
    axis_names = vi.query("is_dense & is_state").index.tolist()
    if has_sparse_states:
        axis_names = ["state_index"] + axis_names

    # lookup_info
    _discrete_states = set(vi.query("is_discrete & is_state").index.tolist())
    lookup_info = {k: v for k, v in model.gridspecs.items() if k in _discrete_states}

    # interpolation info
    _cont_states = set(vi.query("is_continuous & is_state").index.tolist())
    interpolation_info = {k: v for k, v in model.gridspecs.items() if k in _cont_states}

    # indexer infos
    indexer_infos = [
        IndexerInfo(
            axis_names=vi.query("is_sparse & is_state").index.tolist(),
            name="state_indexer",
            out_name="state_index",
        )
    ]

    space_info = SpaceInfo(
        axis_names=axis_names,
        lookup_info=lookup_info,
        interpolation_info=interpolation_info,
        indexer_infos=indexer_infos,
    )

    return state_choice_space, space_info, state_indexers, choice_segments


def create_filter_mask(model, subset, fixed_inputs=None, jit_filter=False):
    """Create mask for combinations of grid values that is True if all filters are True.

    Args:
        model (Model): A processed model.
        subset (list): The subset of variables to be considered in the mask.
        jit_filter (bool): Whether the aggregated filter function is jitted before
            applying it.


        grids (dict): Dictionary containing a one-dimensional grid for each
            variable that is used as a basis to construct the higher dimensional
            grid.
        filters (dict): Dict of filter functions. A filter function depends on
            one or more variables and returns True if a state is feasible.
        fixed_inputs (dict): A dict of fixed inputs for the filters or
            aux_functions. An example would be a model period.


    Returns:
        jax.numpy.ndarray: Multi-Dimensional boolean array that is True
            for a feasible combination of variables. The order of the
            dimensions in the mask is defined by the order of `grids`.

    """
    # preparations
    if subset is None:
        subset = model.variable_info.query("is_sparse").index.tolist()

    fixed_inputs = {} if fixed_inputs is None else fixed_inputs
    _axis_names = [name for name in model.grids if name in subset]
    _filter_names = model.function_info.query("is_filter").index.tolist()

    # Create scalar dag function to evaluate all filters
    _scalar_filter = concatenate_functions(
        functions=model.functions,
        targets=_filter_names,
        aggregator=jnp.logical_and,
    )

    # Apply dispatcher to get mask
    _filter = productmap(_scalar_filter, variables=_axis_names)

    _valid_args = set(inspect.signature(_filter).parameters.keys())
    _potential_kwargs = {**model.grids, **fixed_inputs}

    kwargs = {k: v for k, v in _potential_kwargs.items() if k in _valid_args}

    # Calculate mask
    if jit_filter:
        _filter = jax.jit(_filter)
    mask = _filter(**kwargs)

    return mask


def create_forward_mask(
    initial, grids, next_functions, fixed_inputs=None, aux_functions=None, jit_next=True
):
    """Create a mask for combinations of grid values.

    .. warning::
        This function is extremely experimental and probably buggy

    Args:
        intitial (dict): Dict of arrays with valid combinations of variables.
        grids (dict): Dictionary containing a one-dimensional grid for each
            variable that is used as a basis to construct the higher dimensional
            grid.
        next_functions (dict): Dict of functions for the state space transitions.
            All keys need to start with "next".
        fixed_inputs (dict): A dict of fixed inputs for the next_functions or
            aux_functions. An example would be a model period.
        aux_functions (dict): Auxiliary functions that calculate derived variables
            needed in the filters.
        jit_next (bool): Whether the aggregated next_function is jitted before
            applying it.

    """
    # preparations
    _state_vars = [
        name for name in grids if f"next_{name}" in next_functions
    ]  # sort in order of grids
    _aux_functions = {} if aux_functions is None else aux_functions
    _shape = tuple(len(grids[name]) for name in _state_vars)
    _next_functions = {
        f"next_{name}": next_functions[f"next_{name}"] for name in _state_vars
    }
    _fixed_inputs = {} if fixed_inputs is None else fixed_inputs

    # find valid arguments
    valid_args = set(initial) | set(_aux_functions) | set(_fixed_inputs)

    # find next functions with only valid arguments
    _valid_next_functions = {}
    for name, func in _next_functions.items():
        present_args = set(inspect.signature(func).parameters)
        if present_args.issubset(valid_args):
            _valid_next_functions[name] = func

    # create scalar next function
    _next = concatenate_functions(
        functions={**_valid_next_functions, **_aux_functions},
        targets=list(_valid_next_functions),
        return_type="dict",
    )

    # apply dispatcher
    needed_args = set(inspect.signature(_next).parameters)
    _needed_initial = {k: val for k, val in initial.items() if k in needed_args}
    _gridmapped = spacemap(_next, dense_vars=[], sparse_vars=list(_needed_initial))

    # calculate next values
    if jit_next:
        _gridmapped = jax.jit(_gridmapped)
    _next_values = _gridmapped(**_needed_initial, **_fixed_inputs)

    # create all-false mask
    mask = np.full(_shape, False)

    # fill with full slices to get indexers
    indices = []
    for i, var in enumerate(_state_vars):
        name = f"next_{var}"
        if name in _next_values:
            indices.append(_next_values[name])
        else:
            indices.append(slice(0, _shape[i]))

    # set mask to True with indexers
    mask[tuple(indices)] = True

    return mask


def create_combination_grid(grids, masks, subset=None):
    """Create a grid of all feasible combinations of variables.

    Args:
        grids (dict): Dictionary containing a one-dimensional grid for each
            dimension of the combination grid.
        masks (list): List of masks that define the feasible combinations.
        subset (list): The subset of the variables that enter the combination grid.
            By default all variables in grids are considered.

    Returns:
        dict: Dictionary containing a one-dimensional array for each variable in the
            combination grid. Together these arrays store all feasible combinations
            of variables.

    """
    _subset = list(grids) if subset is None else subset
    _axis_names = [name for name in grids if name in _subset]
    _grids = {name: jnp.array(grids[name]) for name in _axis_names}

    # get combined mask
    _mask_np = np.array(_combine_masks(masks))

    # Calculate meshgrid
    _all_combis = jnp.meshgrid(*_grids.values(), indexing="ij")

    # Flatten meshgrid entries
    combi_grid = {name: arr[_mask_np] for name, arr in zip(_axis_names, _all_combis)}

    return combi_grid


def _combine_masks(masks):
    """Combine multiple masks into one.

    Args:
        masks (list): List of masks.

    Returns:
        np.ndarray: Combined mask.

    """
    if isinstance(masks, (np.ndarray, jnp.ndarray)):
        _masks = [masks]
    else:
        _masks = sorted(masks, key=lambda x: len(x.shape), reverse=True)

    mask = _masks[0]
    for m in _masks[1:]:
        _shape = tuple(list(m.shape) + [1] * (mask.ndim - m.ndim))
        mask = jnp.logical_and(mask, m.reshape(_shape))
    mask = np.array(mask)
    return mask


def create_indexers_and_segments(mask, n_sparse_states, fill_value=-1):
    """Create indexers and segment info related to sparse states and choices.

    Notes:
    ------

    - This probably does not work if there is not at least one sparse state variable
    and at least one sparse choice variable.

    Args:
        mask (np.ndarray): Boolean array with one dimension per state
            or choice variable that is True for feasible state-choice
            combinations. The state variables occupy the first dimensions.
            I.e. the shape is (n_s1, ..., n_sm, n_c1, ..., n_cm).
        n_states (np.ndarray): Number of state variables.
        fill_value (np.ndarray): Value of the index array for infeasible
            states or choices.


    Returns:
        jax.numpy.ndarray: The state indexer with (n_s1, ..., n_sm). The entries are
            ``fill_value`` for infeasible states and count the feasible states
            otherwise.
        jax.numpy.ndarray: the state-choice indexer with shape
            (n_feasible_states, n_c1, ..., n_cn). The entries are ``fill_value`` for
            infeasible state-choice combinations and count the feasible state-choice
            combinations otherwise.

    """
    mask = np.array(mask)

    choice_axes = tuple(range(n_sparse_states, mask.ndim))
    is_feasible_state = mask.any(axis=choice_axes)
    n_feasible_states = np.count_nonzero(is_feasible_state)

    state_indexer = np.full(is_feasible_state.shape, fill_value)
    state_indexer[is_feasible_state] = np.arange(n_feasible_states)

    # reduce mask before doing calculations and using higher dtypes
    reduced_mask = mask[is_feasible_state]

    counter = reduced_mask.cumsum().reshape(reduced_mask.shape) - 1
    state_choice_indexer = np.full(reduced_mask.shape, fill_value)
    state_choice_indexer[reduced_mask] = counter[reduced_mask]

    new_choice_axes = tuple(range(1, mask.ndim - n_sparse_states + 1))
    n_choices = np.count_nonzero(reduced_mask, new_choice_axes)
    segments = np.repeat(np.arange(n_feasible_states), n_choices)

    return (
        jnp.array(state_indexer),
        jnp.array(state_choice_indexer),
        jnp.array(segments),
    )


def _create_value_grid(grids, subset):
    return {name: grid for name, grid in grids.items() if name in subset}
