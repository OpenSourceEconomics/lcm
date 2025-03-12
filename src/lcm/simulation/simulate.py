import logging
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array, vmap

from lcm.dispatchers import simulation_spacemap, vmap_1d
from lcm.interfaces import (
    InternalModel,
    InternalSimulationPeriodResults,
    StateActionSpace,
)
from lcm.max_discrete_actions import get_solve_discrete_problem_policy
from lcm.random import draw_random_seed, generate_simulation_keys
from lcm.simulation.processing import as_panel, process_simulated_data
from lcm.state_action_space import create_state_action_space
from lcm.typing import ParamsDict


def solve_and_simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    compute_ccv_policy_functions: dict[int, Callable[..., tuple[Array, Array]]],
    model: InternalModel,
    next_state: Callable[..., dict[str, Array]],
    logger: logging.Logger,
    solve_model: Callable[..., dict[int, Array]],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """First solve the model and then simulate the model forward in time.

    Same docstring as `simulate` mutatis mutandis.

    """
    vf_arr_dict = solve_model(params)
    return simulate(
        params=params,
        initial_states=initial_states,
        compute_ccv_policy_functions=compute_ccv_policy_functions,
        model=model,
        next_state=next_state,
        logger=logger,
        vf_arr_dict=vf_arr_dict,
        additional_targets=additional_targets,
        seed=seed,
    )


def simulate(
    params: ParamsDict,
    initial_states: dict[str, Array],
    compute_ccv_policy_functions: dict[int, Callable[..., tuple[Array, Array]]],
    model: InternalModel,
    next_state: Callable[..., dict[str, Array]],
    logger: logging.Logger,
    vf_arr_dict: dict[int, Array],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate the model forward in time given pre-computed value function arrays.

    Args:
        params: Dict of model parameters.
        initial_states: List of initial states to start from. Typically from the
            observed dataset.
        compute_ccv_policy_functions: Dict of length n_periods. Each function computes
            the conditional continuation value dependent on the discrete actions.
        next_state: Function that returns the next state given the current
            state and action variables. For stochastic variables, it returns a random
            draw from the distribution of the next state.
        model: Model instance.
        logger: Logger that logs to stdout.
        vf_arr_dict: Dict of value function arrays of length n_periods.
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
    # ----------------------------------------------------------------------------------
    n_periods = len(vf_arr_dict)
    n_initial_states = len(next(iter(initial_states.values())))

    state_action_space = create_state_action_space(
        model=model,
        initial_states=initial_states,
    )

    discrete_policy_calculator = get_solve_discrete_problem_policy(
        variable_info=model.variable_info
    )

    # The following variables are updated during the forward simulation
    states = initial_states
    key = jax.random.key(seed=seed)

    # Forward simulation
    # ----------------------------------------------------------------------------------
    simulation_results = {}

    for period in range(n_periods):
        state_action_space = state_action_space.replace(states)

        # We compute these grid shapes in the loop because they can change over time.
        # TODO (@timmens): This could still be pre-computed in the entry point.  # noqa: TD003,E501
        discrete_actions_grid_shape = tuple(
            len(grid) for grid in state_action_space.discrete_actions.values()
        )
        continuous_actions_grid_shape = tuple(
            len(grid) for grid in state_action_space.continuous_actions.values()
        )

        # Compute optimal continuous action conditional on discrete actions
        # ------------------------------------------------------------------------------
        # We need to pass the value function array of the next period to the continuous
        # action problem solver. If we are at the last period, we pass an empty array.
        next_period_vf_arr = vf_arr_dict.get(period + 1, jnp.empty(0))

        conditional_continuous_action_argmax, conditional_continuous_action_max = (
            solve_continuous_problem(
                data_scs=state_action_space,
                compute_ccv=compute_ccv_policy_functions[period],
                vf_arr=next_period_vf_arr,
                params=params,
            )
        )

        # Get optimal discrete action given the optimal conditional continuous actions
        # ------------------------------------------------------------------------------
        discrete_argmax, action_value = discrete_policy_calculator(
            conditional_continuous_action_max, params=params
        )

        # Get optimal continuous action index given optimal discrete action
        # ------------------------------------------------------------------------------
        continuous_action_argmax = get_continuous_action_argmax_given_discrete(
            conditional_continuous_action_argmax=conditional_continuous_action_argmax,
            discrete_argmax=discrete_argmax,
            discrete_actions_grid_shape=discrete_actions_grid_shape,
        )

        # Convert action indices to action values
        # ------------------------------------------------------------------------------
        discrete_actions = get_values_from_indices(
            flat_indices=discrete_argmax,
            grids=state_action_space.discrete_actions,
            grids_shapes=discrete_actions_grid_shape,
        )

        continuous_actions = get_values_from_indices(
            flat_indices=continuous_action_argmax,
            grids=state_action_space.continuous_actions,
            grids_shapes=continuous_actions_grid_shape,
        )

        # Store results
        # ------------------------------------------------------------------------------
        actions = {**discrete_actions, **continuous_actions}

        simulation_results[period] = InternalSimulationPeriodResults(
            value=action_value,
            actions=actions,
            states=states,
        )

        # Update states
        # ------------------------------------------------------------------------------
        key, stochastic_variables_keys = generate_simulation_keys(
            key=key,
            ids=model.function_info.query("is_stochastic_next").index.tolist(),
        )

        states_with_prefix = next_state(
            **states,
            **actions,
            _period=jnp.repeat(period, n_initial_states),
            params=params,
            keys=stochastic_variables_keys,
        )
        # 'next_' prefix is added by the next_state function, but needs to be removed
        # because in the next period, next states will be current states.
        states = {k.removeprefix("next_"): v for k, v in states_with_prefix.items()}

        logger.info("Period: %s", period)

    processed = process_simulated_data(
        simulation_results,
        model=model,
        params=params,
        additional_targets=additional_targets,
    )

    return as_panel(processed, n_periods=n_periods)


def solve_continuous_problem(
    data_scs: StateActionSpace,
    compute_ccv: Callable[..., tuple[Array, Array]],
    vf_arr: Array,
    params: ParamsDict,
) -> tuple[Array, Array]:
    """Solve the agents' continuous action problem.

    Args:
        data_scs: Class with entries actions and states.
        compute_ccv: Function that returns the conditional continuation
            values for a given combination of states and discrete actions. The function
            depends on:
            - discrete and continuous state variables
            - discrete and continuous action variables
            - vf_arr
            - params
        vf_arr: Value function array.
        params: Dict of model parameters.

    Returns:
        - Jax array with policies for each combination of a state and a discrete action.
          The number and order of dimensions is defined by the `gridmap` function.
        - Jax array with continuation values for each combination of a state and a
          discrete action. The number and order of dimensions is defined by the
          `gridmap` function.

    """
    _gridmapped = simulation_spacemap(
        func=compute_ccv,
        actions_var_names=tuple(data_scs.discrete_actions),
        states_var_names=tuple(data_scs.states),
    )
    gridmapped = jax.jit(_gridmapped)

    return gridmapped(
        **data_scs.states,
        **data_scs.discrete_actions,
        **data_scs.continuous_actions,
        vf_arr=vf_arr,
        params=params,
    )


@partial(vmap_1d, variables=("conditional_continuous_action_argmax", "discrete_argmax"))
def get_continuous_action_argmax_given_discrete(
    conditional_continuous_action_argmax: Array,
    discrete_argmax: Array,
    discrete_actions_grid_shape: tuple[int, ...],
) -> Array:
    """Select optimal continuous action index given optimal discrete action.

    Args:
        conditional_continuous_action_argmax: Index array of optimal continous actions
            conditional on discrete actions.
        discrete_argmax: Index array of optimal discrete actions.
        discrete_actions_grid_shape: Shape of the discrete actions grid.

    Returns:
        Index array of optimal continuous actions.

    """
    indices = jnp.unravel_index(discrete_argmax, shape=discrete_actions_grid_shape)
    return conditional_continuous_action_argmax[indices]


def get_values_from_indices(
    flat_indices: Array,
    grids: dict[str, Array],
    grids_shapes: tuple[int, ...],
) -> dict[str, Array]:
    """Retrieve values from indices.

    Args:
        flat_indices: General indices. Represents the index of the flattened grid.
        grids: Dictionary of grid values.
        grids_shapes: Shape of the grids. Is used to unravel the index.

    Returns:
        Dictionary of values.

    """
    nd_indices = vmapped_unravel_index(flat_indices, grids_shapes)
    return {
        name: grid[index]
        for (name, grid), index in zip(grids.items(), nd_indices, strict=True)
    }


# vmap jnp.unravel_index over the first axis of the `indices` argument, while holding
# the `shape` argument constant (in_axes = (0, None)).
vmapped_unravel_index = vmap(jnp.unravel_index, in_axes=(0, None))
