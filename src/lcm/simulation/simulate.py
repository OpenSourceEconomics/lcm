import logging
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array, vmap

from lcm.discrete_problem import get_solve_discrete_problem_policy
from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.interfaces import InternalModel, StateChoiceSpace
from lcm.random import generate_simulation_keys
from lcm.simulation.processing import as_data_frame, process_simulated_data
from lcm.simulation.state_choice_space import create_state_choice_space
from lcm.typing import ParamsDict
from lcm.utils import draw_random_seed


def solve_and_simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    continuous_choice_grids: dict[int, dict[str, Array]],
    compute_ccv_policy_functions: dict[int, Callable[..., tuple[Array, Array]]],
    model: InternalModel,
    next_state: Callable[..., dict[str, Array]],
    logger: logging.Logger,
    solve_model: Callable[..., list[Array]],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """First solve the model and then simulate the model forward in time.

    Same docstring as `simulate` mutatis mutandis.

    """
    vf_arr_list = solve_model(params)
    return simulate(
        params=params,
        initial_states=initial_states,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_policy_functions=compute_ccv_policy_functions,
        model=model,
        next_state=next_state,
        logger=logger,
        vf_arr_list=vf_arr_list,
        additional_targets=additional_targets,
        seed=seed,
    )


def simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    continuous_choice_grids: dict[int, dict[str, Array]],
    compute_ccv_policy_functions: dict[int, Callable[..., tuple[Array, Array]]],
    model: InternalModel,
    next_state: Callable[..., dict[str, Array]],
    logger: logging.Logger,
    vf_arr_list: list[Array],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        params: Dict of model parameters.
        initial_states: List of initial states to start from. Typically from the
            observed dataset.
        continuous_choice_grids: Dict of length n_periods. Each dict contains 1d grids
            for continuous choice variables.
        compute_ccv_policy_functions: Dict of length n_periods. Each function computes
            the conditional continuation value dependent on the discrete choices.
        next_state: Function that returns the next state given the current
            state and choice variables. For stochastic variables, it returns a random
            draw from the distribution of the next state.
        model: Model instance.
        logger: Logger that logs to stdout.
        vf_arr_list: List of value function arrays of length n_periods.
        additional_targets: List of targets to compute. If provided, the targets
            are computed and added to the simulation results.
        seed: Random number seed; will be passed to `jax.random.key`. If not provided,
            a random seed will be generated.

    Returns:
        DataFrame with the simulation results.

    """
    if seed is None:
        seed = draw_random_seed()

    logger.info("Starting simulation")

    # Preparations
    # ==================================================================================
    n_periods = len(vf_arr_list)
    n_initial_states = len(next(iter(initial_states.values())))

    data_scs = create_state_choice_space(
        model=model,
        initial_states=initial_states,
    )

    # We drop the value function array for the first period, because it is not needed
    # for the simulation. This is because in the first period the agents only consider
    # the current utility and the value function of next period. Similarly, the last
    # value function array is not required, as the agents only consider the current
    # utility in the last period.
    next_vf_arr = dict(
        zip(range(n_periods), vf_arr_list[1:] + [jnp.empty(0)], strict=True)
    )

    discrete_policy_calculator = get_solve_discrete_problem_policy(
        variable_info=model.variable_info,
    )

    # The following variables are updated during the forward simulation
    states = initial_states
    key = jax.random.key(seed=seed)

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
        data_scs = data_scs.replace(states)

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
            vf_arr=next_vf_arr[period],
            params=params,
        )

        # Get optimal discrete choice given the optimal conditional continuous choices
        # ==============================================================================
        discrete_argmax, value = discrete_policy_calculator(ccv, params=params)

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
        key, stochastic_variables_keys = generate_simulation_keys(
            key=key,
            ids=model.function_info.query("is_stochastic_next").index.tolist(),
        )

        states_with_prefix = next_state(
            **states,
            **choices,
            _period=jnp.repeat(period, n_initial_states),
            params=params,
            keys=stochastic_variables_keys,
        )
        # 'next_' prefix is added by the next_state function, but needs to be removed
        # because in the next period, next states will be current states.
        states = {k.removeprefix("next_"): v for k, v in states_with_prefix.items()}

        logger.info("Period: %s", period)

    processed = process_simulated_data(
        _simulation_results,
        model=model,
        params=params,
        additional_targets=additional_targets,
    )

    return as_data_frame(processed, n_periods=n_periods)


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
        compute_ccv: Function that returns the conditional continuation
            values for a given combination of states and discrete choices. The function
            depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - params
        continuous_choice_grids: List of dicts with 1d grids for continuous
            choice variables.
        vf_arr: Value function array.
        params: Dict of model parameters.

    Returns:
        - Jax array with policies for each combination of a state and a discrete choice.
          The number and order of dimensions is defined by the `gridmap` function.
        - Jax array with continuation values for each combination of a state and a
          discrete choice. The number and order of dimensions is defined by the
          `gridmap` function.

    """
    _gridmapped = simulation_spacemap(
        func=compute_ccv,
        choices_var_names=tuple(data_scs.choices),
        states_var_names=tuple(data_scs.states),
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
