import inspect
from functools import partial

import jax.numpy as jnp
from dags import concatenate_functions

from lcm.argmax import argmax, segment_argmax
from lcm.dispatchers import spacemap, vmap_1d
from lcm.interfaces import Space

# ======================================================================================
# Simulate
# ======================================================================================


def simulate(
    params,
    state_indexers,
    continuous_choice_grids,
    compute_ccv_argmax_functions,
    model,
    next_state,
    # output from solution
    vf_arr_list,
    # input to simulate
    initial_states,
):
    """Simulate the model forward in time.

    Args:
        params (dict): Dict of model parameters.
        state_indexers: ...
        continuous_choice_grids (list): List of dicts with 1d grids for continuous
            choice variables.
        compute_ccv_argmax_functions (list): List of functions that compute the
            conditional continuation value dependent on the discrete choices.
        model (Model): Model instance.
        next_state (callable): Function that returns the next state given the current
            state and choice variables.
        vf_arr_list (list): List of value function arrays for each period. Is the output
            of the solution.
        initial_states (list): List of initial states from which we iterate.

    Returns:
        list: List of optimal choices for each initial state per period.

    """
    # Preparations
    # ==================================================================================
    next_state = partial(next_state, params=params)

    n_periods = len(vf_arr_list)

    _discrete_arg_emax_calculator = get_discrete_arg_emax_calculator(
        variable_info=model.variable_info,
    )

    sparse_choice_variables = model.variable_info.query("is_choice & is_sparse").index

    states = initial_states

    # Forward simulation
    # ==================================================================================
    result = []

    for period in range(n_periods):
        # Create data state choice space
        # ------------------------------------------------------------------------------
        # Initial states are treated as sparse variables, so that the sparse variables
        # in the data-state-choice-space correspond to the feasible product of sparse
        # choice variables and initial states. The space has to be created in each
        # iteration because the states change over time.
        # ==============================================================================
        data_state_choice_space, data_choice_segments = create_data_state_choice_space(
            states=states,
            model=model,
        )

        # Compute quantities dependent on data-state-choice-space
        # ==============================================================================
        dense_vars_grid_shape = tuple(
            len(grid) for grid in data_state_choice_space.dense_vars.values()
        )
        cont_choice_grid_shape = tuple(
            len(grid) for grid in continuous_choice_grids[period].values()
        )

        discrete_arg_emax_calculator = partial(
            _discrete_arg_emax_calculator,
            choice_segments=data_choice_segments,
        )

        gridmapped = spacemap(
            func=compute_ccv_argmax_functions[period],
            dense_vars=list(data_state_choice_space.dense_vars),
            sparse_vars=list(data_state_choice_space.sparse_vars),
            dense_first=False,
        )

        # Compute optimal continuous choice conditional on discrete choices
        # ==============================================================================
        conditional_cont_choice_argmax, conditional_continuation_value = gridmapped(
            **data_state_choice_space.dense_vars,
            **continuous_choice_grids[period],
            **data_state_choice_space.sparse_vars,
            **state_indexers[period],
            vf_arr=vf_arr_list[period],
            params=params,
        )

        # Get optimal discrete choice given the optimal conditional continuous choices
        # ==============================================================================
        dense_argmax, sparse_argmax, value = discrete_arg_emax_calculator(
            conditional_continuation_value,
        )

        # Select optimal continuous choice corresponding to optimal discrete choice
        # ==============================================================================
        cont_choice_argmax = select_cont_choice_argmax_given_dense_argmax(
            conditional_cont_choice_argmax,
            dense_argmax=dense_argmax,
            dense_vars_grid_shape=dense_vars_grid_shape,
        )
        if sparse_argmax is not None:
            cont_choice_argmax = cont_choice_argmax[sparse_argmax]

        # Convert optimal choice indices to actual choice values
        # ==============================================================================
        if dense_argmax is None:
            dense_choices = {}
        else:
            dense_choices = retrieve_non_sparse_choices(
                indices=dense_argmax,
                grids=data_state_choice_space.dense_vars,
                grid_shape=dense_vars_grid_shape,
            )

        if cont_choice_argmax is None:
            cont_choices = {}
        else:
            cont_choices = retrieve_non_sparse_choices(
                indices=cont_choice_argmax,
                grids=continuous_choice_grids[period],
                grid_shape=cont_choice_grid_shape,
            )

        sparse_choices = {
            key: data_state_choice_space.sparse_vars[key][sparse_argmax]
            for key in sparse_choice_variables
        }

        # Store results
        # ==============================================================================
        choices = {**dense_choices, **sparse_choices, **cont_choices}
        result.append(
            {
                "choices": choices,
                "value": value,
            },
        )

        # Update states
        # ==============================================================================
        states = next_state(**choices, **states)
        states = {key.lstrip("next_"): val for key, val in states.items()}

    return result


@partial(vmap_1d, variables=["conditional_cont_choice_argmax", "dense_argmax"])
def select_cont_choice_argmax_given_dense_argmax(
    conditional_cont_choice_argmax,
    dense_argmax,
    dense_vars_grid_shape,
):
    if dense_argmax is None:
        out = conditional_cont_choice_argmax
    else:
        indices = jnp.unravel_index(dense_argmax, shape=dense_vars_grid_shape)
        out = conditional_cont_choice_argmax[indices]
    return out


@partial(vmap_1d, variables=["indices"])
def retrieve_non_sparse_choices(indices, grids, grid_shape):
    """Retrieve dense or continuous choices given indices.

    Args:
        indices (int): General index. Represents the index of the flattened grid.
        grids (dict): Dictionary of grids.
        grid_shape (tuple): Shape of the grids. Is used to unravel the index.

    Returns:
        dict: Dictionary of choices.

    """
    indices = jnp.unravel_index(indices, shape=grid_shape)
    return {
        name: grid[index]
        for (name, grid), index in zip(grids.items(), indices, strict=True)
    }


# ======================================================================================
# Data State Choice Space
# ======================================================================================


def create_data_state_choice_space(
    states,
    model,
):
    # preparations
    # ==================================================================================
    vi = model.variable_info

    has_sparse_choice_vars = len(vi.query("is_sparse & is_choice")) > 0

    n_states = len(list(states.values())[0])

    # check that all states have an initial value
    # ==================================================================================
    state_names = set(vi.query("is_state").index)

    if state_names != set(states.keys()):
        raise ValueError(
            "You need to provide an initial value for each state variable in the model."
            f" Missing initial states: {state_names - set(states.keys())}",
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
        for name, state in states.items():
            _combination_grid[name] = jnp.repeat(
                state,
                repeats=n_sc_product_combinations,
            )

        for name, choice in sc_product.items():
            _combination_grid[name] = jnp.tile(choice, reps=n_states)

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
        combination_grid = states
        data_choice_segments = None

    data_state_choice_space = Space(
        sparse_vars=combination_grid,
        dense_vars=dense_choices,
    )

    # create choice segments
    # ==================================================================================
    if has_sparse_choice_vars:
        data_choice_segments = create_choice_segments(
            mask=mask,
            n_sparse_states=n_states,
        )
    else:
        data_choice_segments = None

    return data_state_choice_space, data_choice_segments


# ======================================================================================
# Discrete arg emax
# ======================================================================================


def get_discrete_arg_emax_calculator(variable_info):
    """Return a function that calculates the argmax and max of continuation values.

    The argmax is taken over the discrete choice variables in each state.

    Args:
        variable_info (pd.DataFrame): DataFrame with information about the model
            variables.

    Returns:
        callable: Function that calculates the argmax of the conditional continuation
            values. The function depends on:
            - values (jax.numpy.ndarray): Multidimensional jax array with conditional
                continuation values.
            - choice_segments (jax.numpy.ndarray): Jax array with the indices of the
                choice segments that indicate which sparse choice variables belong to
                one state.

    """
    choice_axes = determine_discrete_dense_choice_axes(variable_info)

    def _calculate_discrete_argmax(values, choice_axes, choice_segments):
        _max = values

        # Determine argmax and max over dense choices
        # ==============================================================================
        if choice_axes is not None:
            dense_argmax, _max = argmax(_max, axis=choice_axes)
        else:
            dense_argmax = None

        # Determine argmax and max over sparse choices
        # ==============================================================================
        if choice_segments is not None:
            sparse_argmax, _max = segment_argmax(_max, **choice_segments)
        else:
            sparse_argmax = None

        return dense_argmax, sparse_argmax, _max

    return partial(_calculate_discrete_argmax, choice_axes=choice_axes)


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
        n_sparse_states (np.ndarray): Number of sparse states

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


def determine_discrete_dense_choice_axes(variable_info):
    """Determine which axes correspond to discrete and dense choices.

    Args:
        variable_info (pd.DataFrame): DataFrame with information about the variables.

    Returns:
        tuple: Tuple of ints, specifying which axes in a value function correspond to
            discrete and dense choices.

    """
    dense_vars = variable_info.query(
        "is_dense & ~(is_choice & is_continuous)",
    ).index.tolist()

    choice_vars = set(variable_info.query("is_choice").index.tolist())

    choice_indices = []
    for i, ax in enumerate(dense_vars):
        if ax in choice_vars:
            choice_indices.append(i)

    choice_indices = None if not choice_indices else tuple(choice_indices)

    return choice_indices
