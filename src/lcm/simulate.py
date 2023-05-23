import inspect
from functools import partial

import jax.numpy as jnp
from dags import concatenate_functions

from lcm.argmax import argmax, segment_argmax
from lcm.discrete_emax import _determine_discrete_choice_axes
from lcm.dispatchers import spacemap, vmap_1d
from lcm.interfaces import Space
from lcm.state_space import create_indexers_and_segments

# ======================================================================================
# Simulate
# ======================================================================================


def simulate(
    params,
    state_indexers,
    continuous_choice_grids,
    compute_ccv_argmax_functions,
    model,
    # output from solution
    vf_arr_list,
    # input to simulate
    initial_states,
):
    data_state_choice_space, data_choice_segments = create_data_state_choice_space(
        initial_states=initial_states,
        model=model,
    )

    # extract information
    n_periods = len(vf_arr_list)

    # container
    optimal_choices = []

    compute_discrete_argmax = get_compute_discrete_argmax(
        variable_info=model.variable_info,
    )

    # forward loop
    for period in range(n_periods):
        gridmapped = spacemap(
            func=compute_ccv_argmax_functions[period],
            dense_vars=list(data_state_choice_space.dense_vars),
            sparse_vars=list(data_state_choice_space.sparse_vars),
            dense_first=False,
        )

        choice, choice_value = gridmapped(
            **data_state_choice_space.dense_vars,
            **continuous_choice_grids[period],
            **data_state_choice_space.sparse_vars,
            **state_indexers[period],
            vf_arr=vf_arr_list[period],
            params=params,
        )

        calculator = partial(
            compute_discrete_argmax,
            choice_segments=data_choice_segments,
        )
        dense_argmax, sparse_argmax, discrete_value = calculator(choice_value)

    return optimal_choices


# ======================================================================================
# Data State Choice Space
# ======================================================================================


def create_data_state_choice_space(
    initial_states,
    model,
):
    # preparations
    # ==================================================================================
    vi = model.variable_info

    has_sparse_choice_vars = len(vi.query("is_sparse & is_choice")) > 0

    n_initial_states = len(list(initial_states.values())[0])

    # check that all states have an initial value
    # ==================================================================================
    state_names = set(vi.query("is_state").index)

    if state_names != set(initial_states.keys()):
        raise ValueError(
            "You need to provide an initial value for each state variable in the model."
            f" Missing initial states: {state_names - set(initial_states.keys())}",
        )

    # get sparse and dense choices
    # ==================================================================================
    sparse_choices = {
        name: grid
        for name, grid in model.grids.items()
        if name in vi.query("is_sparse & is_choice").index.tolist()
    }

    dense_choices = {
        name: grid
        for name, grid in model.grids.items()
        if name in vi.query("is_dense & is_choice & ~is_continuous").index.tolist()
    }

    # create sparse choice state product
    # ==================================================================================
    if has_sparse_choice_vars:
        # create sparse choice product
        # ==============================================================================
        sc_product, n_sc_product_combinations = dict_product(sparse_choices)

        # create full sparse choice state product
        # ==============================================================================
        _combination_grid = {}
        for name, state in initial_states.items():
            _combination_grid[name] = jnp.repeat(
                state,
                repeats=n_sc_product_combinations,
            )

        for name, choice in sc_product.items():
            _combination_grid[name] = jnp.tile(choice, reps=n_initial_states)

        # create filter mask
        # ==============================================================================
        filter_names = model.function_info.query("is_filter").index.tolist()

        scalar_filter = concatenate_functions(
            functions=model.functions,
            targets=filter_names,
            aggregator=jnp.logical_and,
        )

        parameters = list(inspect.signature(scalar_filter).parameters)
        kwargs = {k: v for k, v in _combination_grid.items() if k in parameters}

        _filter = vmap_1d(scalar_filter, variables=parameters)
        mask = _filter(**kwargs)

        # filter infeasible combinations
        # ==============================================================================
        combination_grid = {
            name: grid[mask] for name, grid in _combination_grid.items()
        }

    else:
        combination_grid = initial_states
        data_choice_segments = None

    data_state_choice_space = Space(
        sparse_vars=combination_grid,
        dense_vars=dense_choices,
    )

    # create choice segments
    # ==================================================================================
    if has_sparse_choice_vars:
        _, _, data_choice_segments = create_indexers_and_segments(
            mask=mask,
            n_sparse_states=len(initial_states),
        )
        data_choice_segments = create_choice_segments(
            mask=mask,
            n_sparse_states=len(initial_states),
        )
    else:
        data_choice_segments = None

    return data_state_choice_space, data_choice_segments


# ======================================================================================
# Discrete argmax
# ======================================================================================


def get_compute_discrete_argmax(variable_info):
    choice_axes = tuple(_determine_discrete_choice_axes(variable_info))

    def _calculate_emax_no_shocks(values, choice_axes, choice_segments):
        _max = values

        # find maximum over dense choices
        if choice_axes is not None:
            dense_argmax, _max = argmax(_max, axis=choice_axes)
        else:
            dense_argmax = None

        # find maxmimum over sparse choices
        if choice_segments is not None:
            sparse_argmax, _max = segment_argmax(_max, choice_segments)
        else:
            sparse_argmax = None

        return dense_argmax, sparse_argmax, _max

    return partial(_calculate_emax_no_shocks, choice_axes=choice_axes)


# ======================================================================================
# Next state
# ======================================================================================


def get_next_state_function(model):
    """Combine the next state functions into one function.

    Args:
        model (Model): Model instance.

    Returns:
        function: Combined next state function.

    """
    targets = model.function_info.query("is_next").index.tolist()

    return concatenate_functions(
        functions=model.functions,
        targets=targets,
        return_type="dict",
    )


# ======================================================================================
# Auxiliary
# ======================================================================================


def dict_product(d):
    """Create a product of the entries of a dictionary.

    Args:
        d (dict): Dictionary where all values are arrays.

    Returns:
        - dict: Dictionary with same keys but values correspond to rows of product.
        - int: Number of all combinations.

    """
    arrays = list(d.values())
    grid = jnp.meshgrid(*arrays, indexing="ij")
    stacked = jnp.stack(grid, axis=-1).reshape(-1, len(arrays))
    return dict(zip(d.keys(), list(stacked.T), strict=True)), len(stacked)


def create_choice_segments(mask, n_sparse_states):
    """Create choice segment info related to sparse states and choices.

    Comment: Can be made more memory efficient by reshaping mask into 2d.

    Args:
        mask (jnp.ndarray): Boolean 1d array, where each entry corresponds to a
            data-state-choice combination.
        n_sparse_states (np.ndarray): Number of sparse state variables.

    Returns:
        dict: Dict with segment_info.

    """
    n_choice_combinations = len(mask) // n_sparse_states
    state_ids = jnp.repeat(
        jnp.arange(n_sparse_states),
        repeats=n_choice_combinations,
    )
    segments = state_ids[mask]
    return {
        "segment_ids": jnp.array(segments),
        "num_segments": len(jnp.unique(segments)),
    }
