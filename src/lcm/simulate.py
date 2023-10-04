import inspect
from functools import partial

import jax.numpy as jnp
import pandas as pd
from dags import concatenate_functions
from pybaum import leaf_names, tree_flatten

from lcm.argmax import argmax, segment_argmax
from lcm.dispatchers import spacemap, vmap_1d
from lcm.interfaces import Space
from lcm.model_functions import get_multiply_weights, get_next_weights_function
import jax


def simulate(
    params,
    state_indexers,
    continuous_choice_grids,
    compute_ccv_policy_functions,
    model,
    initial_states,
    solve_model=None,
    vf_arr_list=None,
    additional_targets=None,
    seed=12345,
):
    """Simulate the model forward in time.

    Args:
        params (dict): Dict of model parameters.
        state_indexers (list): List of dicts of length n_periods. Each dict contains one
            or several state indexers.
        continuous_choice_grids (list): List of dicts of length n_periods. Each dict
            contains 1d grids for continuous choice variables.
        compute_ccv_policy_functions (list): List of functions of length n_periods. Each
            function computes the conditional continuation value dependent on the
            discrete choices.
        model (Model): Model instance.
        initial_states (list): List of initial states to start from. Typically from the
            observed dataset.
        solve_model (callable): Function that solves the model. Is only required if
            vf_arr_list is not provided.
        vf_arr_list (list): List of value function arrays of length n_periods. This is
            the output of the model's `solve` function. If not provided, the model is
            solved first.
        additional_targets (list): List of targets to compute. If provided, the targets
            are computed and added to the simulation results.

    Returns:
        list: List of length n_periods containing the valuations, optimal choices, and
            states.

    """
    if vf_arr_list is None:
        if solve_model is None:
            raise ValueError(
                "You need to provide either vf_arr_list or solve_model.",
            )
        vf_arr_list = solve_model(params)

    # Update the vf_arr_list
    # ----------------------------------------------------------------------------------
    # We drop the value function array for the first period, because it is not needed
    # for the simulation. This is because in the first period the agents only consider
    # the current utility and the value function of next period. Similarly, the last
    # value function array is not required, as the agents only consider the current
    # utility in the last period.
    # ==================================================================================
    vf_arr_list = vf_arr_list[1:] + [None]

    # Preparations
    # ==================================================================================
    next_deterministic_states = get_next_deterministic_states_function(model)
    next_deterministic_states = partial(
        next_deterministic_states,
        params=params,
    )

    next_weights = get_next_weights_function(model)
    
    stochastic_variables = model.variable_info.query("is_stochastic").index.tolist()
    multiply_weights = get_multiply_weights(
        stochastic_variables=stochastic_variables,
        vmap_over_first_axis=True,
    )

    n_periods = len(vf_arr_list)

    _discrete_policy_calculator = get_discrete_policy_calculator(
        variable_info=model.variable_info,
    )

    sparse_choice_variables = model.variable_info.query("is_choice & is_sparse").index

    states = initial_states  # will be updated in the forward simulation
    
    key = jax.random.PRNGKey(seed=seed)

    # Forward simulation
    # ==================================================================================
    _simulation_results = []

    for period in range(n_periods):
        # Create data state choice space
        # ------------------------------------------------------------------------------
        # Initial states are treated as sparse variables, so that the sparse variables
        # in the data-state-choice-space correspond to the feasible product of sparse
        # choice variables and initial states. The space has to be created in each
        # iteration because the states change over time.
        # ==============================================================================
        data_scs, data_choice_segments = create_data_scs(
            states=states,
            model=model,
        )

        # Compute objects dependent on data-state-choice-space
        # ==============================================================================
        dense_vars_grid_shape = tuple(
            len(grid) for grid in data_scs.dense_vars.values()
        )
        cont_choice_grid_shape = tuple(
            len(grid) for grid in continuous_choice_grids[period].values()
        )

        discrete_policy_calculator = partial(
            _discrete_policy_calculator,
            choice_segments=data_choice_segments,
        )

        gridmapped = spacemap(
            func=compute_ccv_policy_functions[period],
            dense_vars=list(data_scs.dense_vars),
            sparse_vars=list(data_scs.sparse_vars),
            dense_first=False,
        )

        # Compute optimal continuous choice conditional on discrete choices
        # ==============================================================================
        ccv_policy, ccv = gridmapped(
            **data_scs.dense_vars,
            **continuous_choice_grids[period],
            **data_scs.sparse_vars,
            **state_indexers[period],
            vf_arr=vf_arr_list[period],
            params=params,
        )

        # Get optimal discrete choice given the optimal conditional continuous choices
        # ==============================================================================
        dense_argmax, sparse_argmax, value = discrete_policy_calculator(ccv)

        # Select optimal continuous choice corresponding to optimal discrete choice
        # ------------------------------------------------------------------------------
        # The conditional continuous choice argmax is computed for each discrete choice
        # in the data-state-choice-space. Here we select the the optimal continuous
        # choice corresponding to the optimal discrete choice (dense and sparse).
        # ==============================================================================
        cont_choice_argmax = filter_ccv_policy(
            ccv_policy,
            dense_argmax=dense_argmax,
            dense_vars_grid_shape=dense_vars_grid_shape,
        )
        if sparse_argmax is not None:
            cont_choice_argmax = cont_choice_argmax[sparse_argmax]

        # Convert optimal choice indices to actual choice values
        # ==============================================================================
        dense_choices = retrieve_non_sparse_choices(
            indices=dense_argmax,
            grids=data_scs.dense_vars,
            grid_shape=dense_vars_grid_shape,
        )

        cont_choices = retrieve_non_sparse_choices(
            indices=cont_choice_argmax,
            grids=continuous_choice_grids[period],
            grid_shape=cont_choice_grid_shape,
        )

        sparse_choices = {
            key: data_scs.sparse_vars[key][sparse_argmax]
            for key in sparse_choice_variables
        }

        # Store results
        # ==============================================================================
        choices = {**dense_choices, **sparse_choices, **cont_choices}

        _simulation_results.append(
            {
                "value": value,
                "choices": choices,
                "states": states,
            },
        )

        # Update states
        # ==============================================================================
        deterministic_states = next_deterministic_states(**choices, **states)

        weights = next_weights(**states, params=params)
        node_weights = multiply_weights(**weights)
        
        key, sim_key = jax.random.split(key)

        stochastic_states = _draw_stochastic_states(
            node_weights, grids=model.grids, stochastic_variables=stochastic_variables, key=sim_key
        )

        states = {**deterministic_states, **stochastic_states}
        states = {key.removeprefix("next_"): val for key, val in states.items()}

    processed = _process_simulated_data(_simulation_results)

    if additional_targets is not None:
        calculated_targets = _compute_targets(
            processed,
            targets=additional_targets,
            model_functions=model.functions,
            params=params,
        )
        processed = {**processed, **calculated_targets}

    return _as_data_frame(processed, n_periods=n_periods)


def _draw_stochastic_states(weights, grids, stochastic_variables, key):
    def _random_choice(key, probs, labels):
        return jax.random.choice(key, labels, p=probs)
    
    keys = jax.random.split(key, weights.shape[0])
    
    random_choice = jax.vmap(_random_choice, in_axes=(0, 0, None))
    
    name = stochastic_variables[0]
    
    return {name: random_choice(keys, weights, grids[name])}



def _as_data_frame(processed, n_periods):
    """Convert processed simulation results to DataFrame.

    Args:
        processed (dict): Dict with processed simulation results.
        n_periods (int): Number of periods.

    Returns:
        pd.DataFrame: DataFrame with the simulation results. The index is a multi-index
            with the first level corresponding to the period and the second level
            corresponding to the initial state id. The columns correspond to the value,
            and the choice and state variables, and potentially auxiliary variables.

    """
    n_initial_states = len(processed["value"]) // n_periods
    index = pd.MultiIndex.from_product(
        [range(n_periods), range(n_initial_states)],
        names=["period", "initial_state_id"],
    )
    return pd.DataFrame(processed, index=index)


def _compute_targets(processed_results, targets, model_functions, params):
    """Compute targets.

    Args:
        processed_results (dict): Dict with processed simulation results. Values must be
            one-dimensional arrays.
        targets (list): List of targets to compute.
        model_functions (dict): Dict with model functions.
        params (dict): Dict with model parameters.

    Returns:
        dict: Dict with computed targets.

    """
    target_func = concatenate_functions(
        functions=model_functions,
        targets=targets,
        return_type="dict",
    )

    # get list of variables over which we want to vectorize the target function
    variables = [
        p for p in list(inspect.signature(target_func).parameters) if p != "params"
    ]

    target_func = vmap_1d(target_func, variables=variables)

    kwargs = {k: v for k, v in processed_results.items() if k in variables}
    return target_func(params=params, **kwargs)


def _process_simulated_data(results):
    """Process and flatten the simulation results.

    Args:
        results (list): List of dicts with simulation results. Each dict contains the
            value, choices, and states for one period.

    Returns:
        dict: Dict with processed simulation results. The keys are the variable names
            and the values are the flattened arrays, with dimension (n_periods *
            n_initial_states, ).

    """
    column_names = [
        # remove prefixes 'choice_' and 'states_' from variable names, which are added
        # by the leaf_names function
        name.removeprefix("choices_").removeprefix("states_")
        for name in leaf_names(results[0])
    ]

    # ==================================================================================
    # Get dict of arrays for each var with dimension (n_periods * n_initial_states, )
    # ----------------------------------------------------------------------------------
    # The arrays are flattened, so that the resulting dictionary has a one-dimensional
    # array for each variable. The length of this array is the number of periods times
    # the number of initial states. The order of array elements is given by an outer
    # level of periods and an inner level of initial states ids.
    # ==================================================================================

    # flatten the nested dictionary structure to get a list of dicts, where each dict
    # has only array values
    processed = [
        dict(zip(column_names, tree_flatten(vals)[0], strict=True)) for vals in results
    ]
    processed = {key: [d[key] for d in processed] for key in column_names}
    return {key: jnp.concatenate(values) for key, values in processed.items()}


def get_next_deterministic_states_function(model):
    """Get function that returns the deterministic next states.

    Args:
        model (Model): Model instance.

    Returns:
        tuple: Tuple of functions. First function returns the non-stochastic next
            states, second function returns the stochastic next states.

    """
    deterministic_targets = model.function_info.query(
        "is_next & ~is_stochastic_next",
    ).index

    return concatenate_functions(
        functions=model.functions,
        targets=deterministic_targets,
        return_type="dict",
        enforce_signature=False,
    )


@partial(vmap_1d, variables=["ccv_policy", "dense_argmax"])
def filter_ccv_policy(
    ccv_policy,
    dense_argmax,
    dense_vars_grid_shape,
):
    """Select optimal continuous choice index given optimal discrete choice.

    Args:
        ccv_policy (jax.numpy.ndarray): Index array of optimal continous choices
            conditional on discrete choices.
        dense_argmax (jax.numpy.array): Index array of optimal dense choices.
        dense_vars_grid_shape (tuple): Shape of the dense variables grid.

    Returns:
        jax.numpy.ndarray: Index array of optimal continuous choices.

    """
    if dense_argmax is None:
        out = ccv_policy
    else:
        indices = jnp.unravel_index(dense_argmax, shape=dense_vars_grid_shape)
        out = ccv_policy[indices]
    return out


def retrieve_non_sparse_choices(indices, grids, grid_shape):
    """Retrieve dense or continuous choices given indices.

    Args:
        indices (jnp.numpy.ndarray or None): General indices. Represents the index of
            the flattened grid.
        grids (dict): Dictionary of grids.
        grid_shape (tuple): Shape of the grids. Is used to unravel the index.

    Returns:
        dict: Dictionary of choices.

    """
    if indices is None:
        out = {}
    else:
        out = _retrieve_non_sparse_choices(indices, grids, grid_shape)
    return out


@partial(vmap_1d, variables=["index"])
def _retrieve_non_sparse_choices(index, grids, grid_shape):
    """Retrieve dense or continuous choices given index.

    Args:
        index (int): General index. Represents the index of the flattened grid.
        grids (dict): Dictionary of grids.
        grid_shape (tuple): Shape of the grids. Is used to unravel the index.

    Returns:
        dict: Dictionary of choices.

    """
    indices = jnp.unravel_index(index, shape=grid_shape)
    return {
        name: grid[index]
        for (name, grid), index in zip(grids.items(), indices, strict=True)
    }


# ======================================================================================
# Data State Choice Space
# ======================================================================================


def create_data_scs(
    states,
    model,
):
    """Create data state choice space.

    Args:
        states (dict): Dict with initial states.
        model (Model): Model instance.

    Returns:
        - Space: Data state choice space.
        - dict: Dict with choice segments.

    """
    # preparations
    # ==================================================================================
    vi = model.variable_info

    has_sparse_choice_vars = len(vi.query("is_sparse & is_choice")) > 0

    n_states = len(next(iter(states.values())))

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

    data_scs = Space(
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

    return data_scs, data_choice_segments


# ======================================================================================
# Discrete policy
# ======================================================================================


def get_discrete_policy_calculator(variable_info):
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
    discrete_dense_choice_vars = variable_info.query(
        "~is_continuous & is_dense & is_choice",
    ).index.tolist()

    choice_vars = set(variable_info.query("is_choice").index.tolist())

    # We add 1 because the first dimension corresponds to the sparse state variables
    choice_indices = [
        i + 1 for i, ax in enumerate(discrete_dense_choice_vars) if ax in choice_vars
    ]

    return None if not choice_indices else tuple(choice_indices)
