from collections.abc import Callable
from functools import partial
from typing import Literal

import jax
import pandas as pd
from jax import Array

from lcm.input_processing import process_model
from lcm.interfaces import StateActionSpace, StateSpaceInfo
from lcm.logging import get_logger
from lcm.max_continuous_actions import (
    get_argmax_Q_over_c,
    get_max_Q_over_c,
)
from lcm.max_discrete_actions import get_max_Qc
from lcm.next_state import get_next_state_function
from lcm.simulation.simulate import simulate, solve_and_simulate
from lcm.solution.solve_brute import solve
from lcm.state_action_space import (
    create_state_action_space,
    create_state_space_info,
)
from lcm.typing import (
    ArgmaxQOverCFunction,
    MaxQcFunction,
    MaxQOverCFunction,
    ParamsDict,
    Target,
)
from lcm.user_model import Model
from lcm.utility_and_feasibility import (
    get_utility_and_feasibility_function,
)


def get_lcm_function(
    model: Model,
    *,
    targets: Literal["solve", "simulate", "solve_and_simulate"],
    debug_mode: bool = True,
    jit: bool = True,
) -> tuple[Callable[..., dict[int, Array] | pd.DataFrame], ParamsDict]:
    """Entry point for users to get high level functions generated by lcm.

    Return the function to solve and/or simulate a model along with a template for the
    parameters.

    Advanced users might want to use lower level functions instead, but can read the
    source code of this function to see how the lower level components are meant to be
    used.

    Args:
        model: User model specification.
        targets: The requested function types. Currently only "solve", "simulate" and
            "solve_and_simulate" are supported.
        debug_mode: Whether to log debug messages.
        jit: Whether to jit the returned function.

    Returns:
        - A function that takes params (and possibly other arguments, such as initial
          states in the simulate case) and returns the requested targets.
        - A parameter dictionary where all parameter values are initialized to NaN.

    """
    # ==================================================================================
    # preparations
    # ==================================================================================
    if targets not in {"solve", "simulate", "solve_and_simulate"}:
        raise NotImplementedError

    internal_model = process_model(model)
    last_period = internal_model.n_periods - 1

    logger = get_logger(debug_mode=debug_mode)

    # ==================================================================================
    # Create model functions and state-action-spaces
    # ==================================================================================
    state_action_spaces: dict[int, StateActionSpace] = {}
    state_space_infos: dict[int, StateSpaceInfo] = {}
    max_Q_over_c_functions: dict[int, MaxQOverCFunction] = {}
    argmax_Q_over_c_functions: dict[int, ArgmaxQOverCFunction] = {}
    max_Qc_functions: dict[int, MaxQcFunction] = {}

    for period in reversed(range(internal_model.n_periods)):
        is_last_period = period == last_period

        state_action_space = create_state_action_space(
            model=internal_model,
            is_last_period=is_last_period,
        )

        state_space_info = create_state_space_info(
            model=internal_model,
            is_last_period=is_last_period,
        )

        if is_last_period:
            next_state_space_info = LastPeriodsNextStateSpaceInfo
        else:
            next_state_space_info = state_space_infos[period + 1]

        u_and_f = get_utility_and_feasibility_function(
            model=internal_model,
            next_state_space_info=next_state_space_info,
            period=period,
            is_last_period=is_last_period,
        )

        max_Q_over_c = get_max_Q_over_c(
            utility_and_feasibility=u_and_f,
            continuous_actions_names=tuple(state_action_space.continuous_actions),
            states_and_discrete_actions_names=state_action_space.states_and_discrete_actions_names,
        )

        argmax_Q_over_c = get_argmax_Q_over_c(
            utility_and_feasibility=u_and_f,
            continuous_actions_names=tuple(state_action_space.continuous_actions),
        )

        max_Qc = get_max_Qc(
            random_utility_shock_type=internal_model.random_utility_shocks,
            variable_info=internal_model.variable_info,
            is_last_period=is_last_period,
        )

        state_action_spaces[period] = state_action_space
        state_space_infos[period] = state_space_info
        max_Q_over_c_functions[period] = max_Q_over_c
        argmax_Q_over_c_functions[period] = argmax_Q_over_c
        max_Qc_functions[period] = max_Qc

    # ==================================================================================
    # select requested solver and partial arguments into it
    # ==================================================================================
    _solve_model = partial(
        solve,
        state_action_spaces=state_action_spaces,
        max_Q_over_c_functions=max_Q_over_c_functions,
        max_Qc_functions=max_Qc_functions,
        logger=logger,
    )
    solve_model = jax.jit(_solve_model) if jit else _solve_model

    _next_state_simulate = get_next_state_function(
        model=internal_model, target=Target.SIMULATE
    )
    next_state_simulate = jax.jit(_next_state_simulate) if jit else _next_state_simulate
    simulate_model = partial(
        simulate,
        argmax_Q_over_c_functions=argmax_Q_over_c_functions,
        model=internal_model,
        next_state=next_state_simulate,  # type: ignore[arg-type]
        logger=logger,
    )

    solve_and_simulate_model = partial(
        solve_and_simulate,
        argmax_Q_over_c_functions=argmax_Q_over_c_functions,
        model=internal_model,
        next_state=next_state_simulate,  # type: ignore[arg-type]
        logger=logger,
        solve_model=solve_model,
    )

    target_func: Callable[..., dict[int, Array] | pd.DataFrame]

    if targets == "solve":
        target_func = solve_model
    elif targets == "simulate":
        target_func = simulate_model
    elif targets == "solve_and_simulate":
        target_func = solve_and_simulate_model

    return target_func, internal_model.params


LastPeriodsNextStateSpaceInfo = StateSpaceInfo(
    states_names=(),
    discrete_states={},
    continuous_states={},
)
