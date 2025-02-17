import inspect
import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
from dags import concatenate_functions
from jax import Array, vmap

from lcm.argmax import argmax
from lcm.dispatchers import spacemap, vmap_1d
from lcm.interfaces import InternalModel, StateChoiceSpace
from lcm.typing import InternalUserFunction, ParamsDict


def simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    continuous_choice_grids: list[dict[str, Array]],
    compute_ccv_policy_functions: list[Callable[..., tuple[Array, Array]]],
    model: InternalModel,
    next_state: Callable[..., dict[str, Array]],
    logger: logging.Logger,
    solve_model: Callable[..., list[Array]] | None = None,
    pre_computed_vf_arr_list: list[Array] | None = None,
    additional_targets: list[str] | None = None,
    seed: int = 12345,
) -> pd.DataFrame:
    """Simulate the model forward in time.

    Args:
        params (dict): Dict of model parameters.
        initial_states (list): List of initial states to start from. Typically from the
            observed dataset.
        continuous_choice_grids (list): List of dicts of length n_periods. Each dict
            contains 1d grids for continuous choice variables.
        compute_ccv_policy_functions (list): List of functions of length n_periods. Each
            function computes the conditional continuation value dependent on the
            discrete choices.
        next_state (callable): Function that returns the next state given the current
            state and choice variables. For stochastic variables, it returns a random
            draw from the distribution of the next state.
        model (Model): Model instance.
        logger (logging.Logger): Logger that logs to stdout.
        solve_model (callable): Function that solves the model. Is only required if
            vf_arr_list is not provided.
        pre_computed_vf_arr_list (list): List of value function arrays of length
            n_periods. This is the output of the model's `solve` function. If not
            provided, the model is solved first.
        additional_targets (list): List of targets to compute. If provided, the targets
            are computed and added to the simulation results.
        seed (int): Random number seed; will be passed to `jax.random.PRNGKey`.

    Returns:
        list: List of length n_periods containing the valuations, optimal choices, and
            states.

    """
    if pre_computed_vf_arr_list is None:
        if solve_model is None:
            raise ValueError(
                "You need to provide either vf_arr_list or solve_model.",
            )
        # We do not need to convert the params here, because the solve_model function
        # will do it.
        vf_arr_list = solve_model(params)
    else:
        vf_arr_list = pre_computed_vf_arr_list

    logger.info("Starting simulation")

    # Update the vf_arr_list
    # ----------------------------------------------------------------------------------
    # We drop the value function array for the first period, because it is not needed
    # for the simulation. This is because in the first period the agents only consider
    # the current utility and the value function of next period. Similarly, the last
    # value function array is not required, as the agents only consider the current
    # utility in the last period.
    # ==================================================================================
    vf_arr_list = vf_arr_list[1:] + [jnp.empty(0)]

    # Preparations
    # ==================================================================================
    n_periods = len(vf_arr_list)
    n_initial_states = len(next(iter(initial_states.values())))

    discrete_policy_calculator = get_discrete_policy_calculator(
        variable_info=model.variable_info,
    )

    # The following variables are updated during the forward simulation
    states = initial_states
    key = jax.random.PRNGKey(seed=seed)

    # Forward simulation
    # ==================================================================================
    _simulation_results = []

    for period in range(n_periods):
        # Create data state choice space
        # ------------------------------------------------------------------------------
        # Initial states are treated as combination variables, so that the combination
        # variables in the data-state-choice-space correspond to the feasible product
        # of combination variables and initial states. The space has to be created in
        # each iteration because the states change over time.
        # ==============================================================================
        data_scs = create_data_scs(
            states=states,
            model=model,
        )

        # Compute objects dependent on data-state-choice-space
        # ==============================================================================
        choices_grid_shape = tuple(len(grid) for grid in data_scs.choices.values())
        cont_choices_grid_shape = tuple(
            len(grid) for grid in continuous_choice_grids[period].values()
        )

        # Compute optimal continuous choice conditional on discrete choices
        # ==============================================================================
        ccv_policy, ccv = solve_continuous_problem(
            data_scs=data_scs,
            compute_ccv=compute_ccv_policy_functions[period],
            continuous_choice_grids=continuous_choice_grids[period],
            vf_arr=vf_arr_list[period],
            params=params,
        )

        # Get optimal discrete choice given the optimal conditional continuous choices
        # ==============================================================================
        discrete_argmax, value = discrete_policy_calculator(ccv)

        # Select optimal continuous choice corresponding to optimal discrete choice
        # ------------------------------------------------------------------------------
        # The conditional continuous choice argmax is computed for each discrete choice
        # in the data-state-choice-space. Here we select the the optimal continuous
        # choice corresponding to the optimal discrete choice.
        # ==============================================================================
        cont_choice_argmax = filter_ccv_policy(
            ccv_policy=ccv_policy,
            discrete_argmax=discrete_argmax,
            vars_grid_shape=choices_grid_shape,
        )

        # Convert optimal choice indices to actual choice values
        # ==============================================================================
        choices = retrieve_choices(
            flat_indices=discrete_argmax,
            grids=data_scs.choices,
            grids_shapes=choices_grid_shape,
        )

        cont_choices = retrieve_choices(
            flat_indices=cont_choice_argmax,
            grids=continuous_choice_grids[period],
            grids_shapes=cont_choices_grid_shape,
        )

        # Store results
        # ==============================================================================
        choices = {**choices, **cont_choices}

        _simulation_results.append(
            {
                "value": value,
                "choices": choices,
                "states": states,
            },
        )

        # Update states
        # ==============================================================================
        key, sim_keys = _generate_simulation_keys(
            key=key,
            ids=model.function_info.query("is_stochastic_next").index.tolist(),
        )

        states = next_state(
            **states,
            **choices,
            _period=jnp.repeat(period, n_initial_states),
            params=params,
            keys=sim_keys,
        )

        # 'next_' prefix is added by the next_state function, but needs to be removed
        # because in the next period, next states are current states.
        states = {k.removeprefix("next_"): v for k, v in states.items()}

        logger.info("Period: %s", period)

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


def solve_continuous_problem(
    data_scs: StateChoiceSpace,
    compute_ccv: Callable[..., tuple[Array, Array]],
    continuous_choice_grids: dict[str, Array],
    vf_arr: Array,
    params: ParamsDict,
) -> tuple[Array, Array]:
    """Solve the agent's continuous choices problem problem.

    Args:
        data_scs: Class with entries choices and states.
        compute_ccv (callable): Function that returns the conditional continuation
            values for a given combination of states and discrete choices. The function
            depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - params
        continuous_choice_grids (list): List of dicts with 1d grids for continuous
            choice variables.
        vf_arr (jax.Array): Value function array.
        params (dict): Dict of model parameters.

    Returns:
        - jnp.ndarray: Jax array with policies for each combination of a state and a
          discrete choice. The number and order of dimensions is defined by the
          `gridmap` function.
        - jnp.ndarray: Jax array with continuation values for each combination of a
            state and a discrete choice. The number and order of dimensions is defined
            by the `gridmap` function.

    """
    _gridmapped = spacemap(
        func=compute_ccv,
        product_vars=tuple(data_scs.choices),
        combination_vars=tuple(data_scs.states),
    )
    gridmapped = jax.jit(_gridmapped)

    return gridmapped(
        **data_scs.choices,
        **data_scs.states,
        **continuous_choice_grids,
        vf_arr=vf_arr,
        params=params,
    )


# ======================================================================================
# Output processing
# ======================================================================================


def _as_data_frame(processed: dict[str, Array], n_periods: int) -> pd.DataFrame:
    """Convert processed simulation results to DataFrame.

    Args:
        processed: Dict with processed simulation results.
        n_periods: Number of periods.

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


def _compute_targets(
    processed_results: dict[str, Array],
    targets: list[str],
    model_functions: dict[str, InternalUserFunction],
    params: ParamsDict,
) -> dict[str, Array]:
    """Compute targets.

    Args:
        processed_results: Dict with processed simulation results. Values must be
            one-dimensional arrays.
        targets: List of targets to compute.
        model_functions: Dict with model functions.
        params: Dict with model parameters.

    Returns:
        dict: Dict with computed targets.

    """
    target_func = concatenate_functions(
        functions=model_functions,
        targets=targets,
        return_type="dict",
    )

    # get list of variables over which we want to vectorize the target function
    variables = tuple(
        p for p in list(inspect.signature(target_func).parameters) if p != "params"
    )

    target_func = vmap_1d(target_func, variables=variables)

    kwargs = {k: v for k, v in processed_results.items() if k in variables}
    return target_func(params=params, **kwargs)


def _process_simulated_data(results: list[dict[str, Any]]) -> dict[str, Array]:
    """Process and flatten the simulation results.

    This function produces a dict of arrays for each var with dimension (n_periods *
    n_initial_states,). The arrays are flattened, so that the resulting dictionary has a
    one-dimensional array for each variable. The length of this array is the number of
    periods times the number of initial states. The order of array elements is given by
    an outer level of periods and an inner level of initial states ids.


    Args:
        results (list): List of dicts with simulation results. Each dict contains the
            value, choices, and states for one period. Choices and states are stored in
            a nested dictionary.

    Returns:
        dict: Dict with processed simulation results. The keys are the variable names
            and the values are the flattened arrays, with dimension (n_periods *
            n_initial_states, ). Additionally, the _period variable is added.

    """
    n_periods = len(results)
    n_initial_states = len(results[0]["value"])

    list_of_dicts = [
        {"value": d["value"], **d["choices"], **d["states"]} for d in results
    ]
    dict_of_lists = {
        key: [d[key] for d in list_of_dicts] for key in list(list_of_dicts[0])
    }
    out = {key: jnp.concatenate(values) for key, values in dict_of_lists.items()}
    out["_period"] = jnp.repeat(jnp.arange(n_periods), n_initial_states)

    return out


# ======================================================================================
# Simulation keys
# ======================================================================================


def _generate_simulation_keys(
    key: Array, ids: list[str]
) -> tuple[Array, dict[str, Array]]:
    """Generate PRNG keys for simulation.

    Args:
        key: PRNG key.
        ids: List of names for which a key is to be generated.

    Returns:
        - Updated PRNG key.
        - Dict with PRNG keys for each id in ids.

    """
    keys = jax.random.split(key, num=len(ids) + 1)

    key = keys[0]
    simulation_keys = dict(zip(ids, keys[1:], strict=True))

    return key, simulation_keys


# ======================================================================================
# Filter policy
# ======================================================================================


@partial(vmap_1d, variables=("ccv_policy", "discrete_argmax"))
def filter_ccv_policy(
    ccv_policy: Array,
    discrete_argmax: Array,
    vars_grid_shape: tuple[int, ...],
) -> Array:
    """Select optimal continuous choice index given optimal discrete choice.

    Args:
        ccv_policy: Index array of optimal continous choices
            conditional on discrete choices.
        discrete_argmax: Index array of optimal discrete choices.
        vars_grid_shape: Shape of the variables grid.

    Returns:
        Index array of optimal continuous choices.

    """
    if discrete_argmax is None:
        out = ccv_policy
    else:
        indices = jnp.unravel_index(discrete_argmax, shape=vars_grid_shape)
        out = ccv_policy[indices]
    return out


def retrieve_choices(
    flat_indices: Array,
    grids: dict[str, Array],
    grids_shapes: tuple[int, ...],
) -> dict[str, Array]:
    """Retrieve choices given flat indices.

    Args:
        flat_indices: General indices. Represents the index of the flattened grid.
        grids: Dictionary of grid values.
        grids_shapes: Shape of the grids. Is used to unravel the index.

    Returns:
        Dictionary of choices.

    """
    nd_indices = vmapped_unravel_index(flat_indices, grids_shapes)
    return {
        name: grid[index]
        for (name, grid), index in zip(grids.items(), nd_indices, strict=True)
    }


# vmap jnp.unravel_index over the first axis of the `indices` argument, while holding
# the `shape` argument constant (in_axes = (0, None)).
vmapped_unravel_index = vmap(jnp.unravel_index, in_axes=(0, None))


# ======================================================================================
# Data State Choice Space
# ======================================================================================


def create_data_scs(
    states: dict[str, Array],
    model: InternalModel,
) -> StateChoiceSpace:
    """Create data state choice space.

    Args:
        states (dict): Dict with initial states.
        model: Model instance.
        period (int): Period.

    Returns:
        - Space: Data state choice space.
        - None

    """
    # preparations
    # ==================================================================================
    vi = model.variable_info

    # check that all states have an initial value
    # ==================================================================================
    state_names = set(vi.query("is_state").index)

    if state_names != set(states.keys()):
        missing = state_names - set(states.keys())
        too_many = set(states.keys()) - state_names
        raise ValueError(
            "You need to provide an initial value for each state variable in the model."
            f"\n\nMissing initial states: {missing}\n",
            f"Provided variables that are not states: {too_many}",
        )

    # get choices
    # ==================================================================================
    choices = {
        name: grid
        for name, grid in model.grids.items()
        if name in vi.query("is_choice & is_discrete").index.tolist()
    }

    return StateChoiceSpace(
        states=states,
        choices=choices,
        ordered_var_names=tuple(vi.query("is_state | is_discrete").index.tolist()),
    )


# ======================================================================================
# Discrete policy
# ======================================================================================


def get_discrete_policy_calculator(
    variable_info: pd.DataFrame,
) -> Callable[..., tuple[Array, Array]]:
    """Return a function that calculates the argmax and max of continuation values.

    The argmax is taken over the discrete choice variables in each state.

    Args:
        variable_info (pd.DataFrame): DataFrame with information about the model
            variables.

    Returns:
        callable: Function that calculates the argmax of the conditional continuation
            values. The function depends on:
            - values (jax.Array): Multidimensional jax array with conditional
                continuation values.

    """
    choice_axes = determine_discrete_choice_axes(variable_info)

    def _calculate_discrete_argmax(
        values: Array, choice_axes: tuple[int, ...]
    ) -> tuple[Array, Array]:
        return argmax(values, axis=choice_axes)

    return partial(_calculate_discrete_argmax, choice_axes=choice_axes)


# ======================================================================================
# Auxiliary
# ======================================================================================


def dict_product(d: dict[str, Array]) -> tuple[dict[str, Array], int]:
    """Create a product of the entries of a dictionary.

    Args:
        d: Dictionary where all values are arrays, and keys are strings.

    Returns:
        - dict: Dictionary with same keys but values correspond to rows of product.
        - int: Number of all combinations.

    """
    arrays = list(d.values())
    grid = jnp.meshgrid(*arrays, indexing="ij")
    stacked = jnp.stack(grid, axis=-1).reshape(-1, len(arrays))
    return dict(zip(d.keys(), list(stacked.T), strict=True)), len(stacked)


def determine_discrete_choice_axes(variable_info: pd.DataFrame) -> tuple[int, ...]:
    """Determine which axes correspond to discrete choices.

    Args:
        variable_info (pd.DataFrame): DataFrame with information about the variables.

    Returns:
        tuple: Tuple of ints, specifying which axes in a value function correspond to
            discrete choices.

    """
    discrete_choice_vars = variable_info.query(
        "is_choice & is_discrete",
    ).index.tolist()

    choice_vars = set(variable_info.query("is_choice").index.tolist())

    # The first dimension corresponds to the simulated states, so add 1.
    return tuple(
        1 + i for i, ax in enumerate(discrete_choice_vars) if ax in choice_vars
    )
