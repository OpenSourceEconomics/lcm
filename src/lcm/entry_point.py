import functools
from collections.abc import Callable
from functools import partial
from typing import Literal, cast

import jax
import jax.numpy as jnp

from lcm.argmax import argmax
from lcm.discrete_problem import get_solve_discrete_problem
from lcm.dispatchers import productmap
from lcm.input_processing import process_model
from lcm.logging import get_logger
from lcm.model_functions import (
    get_utility_and_feasibility_function,
)
from lcm.next_state import get_next_state_function
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve
from lcm.solution.state_space import create_state_choice_space
from lcm.typing import ParamsDict
from lcm.user_model import Model


def get_lcm_function(
    model: Model,
    targets: Literal["solve", "simulate", "solve_and_simulate"] = "solve",
    *,
    debug_mode: bool = True,
    jit: bool = True,
) -> tuple[Callable, ParamsDict]:
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

    _mod = process_model(model)
    last_period = _mod.n_periods - 1

    logger = get_logger(debug_mode)

    # ==================================================================================
    # create list of continuous choice grids
    # ==================================================================================
    # for now they are the same in all periods but this will change.
    _subset = _mod.variable_info.query("is_continuous & is_choice").index.tolist()
    _choice_grids = {k: _mod.grids[k] for k in _subset}
    continuous_choice_grids = [_choice_grids] * _mod.n_periods

    # ==================================================================================
    # Initialize other argument lists
    # ==================================================================================
    state_choice_spaces = []
    space_infos = []
    compute_ccv_functions = []
    compute_ccv_policy_functions = []
    choice_segments = []  # type: ignore[var-annotated]
    emax_calculators = []

    # ==================================================================================
    # Create stace choice space for each period
    # ==================================================================================
    for period in range(_mod.n_periods):
        is_last_period = period == last_period

        # call state space creation function, append trivial items to their lists
        # ==============================================================================
        sc_space, space_info = create_state_choice_space(
            model=_mod,
            is_last_period=is_last_period,
        )

        state_choice_spaces.append(sc_space)
        choice_segments.append(None)
        space_infos.append(space_info)

    # ==================================================================================
    # Shift space info (in period t we require the space info of period t+1)
    # ==================================================================================
    space_infos = space_infos[1:] + [{}]  # type: ignore[list-item]

    # ==================================================================================
    # Create model functions
    # ==================================================================================
    for period in range(_mod.n_periods):
        is_last_period = period == last_period

        # create the compute conditional continuation value functions and append to list
        # ==============================================================================
        u_and_f = get_utility_and_feasibility_function(
            model=_mod,
            space_info=space_infos[period],
            name_of_values_on_grid="vf_arr",
            period=period,
            is_last_period=is_last_period,
        )

        compute_ccv = create_compute_conditional_continuation_value(
            utility_and_feasibility=u_and_f,
            continuous_choice_variables=list(_choice_grids),
        )
        compute_ccv_functions.append(compute_ccv)

        compute_ccv_argmax = create_compute_conditional_continuation_policy(
            utility_and_feasibility=u_and_f,
            continuous_choice_variables=list(_choice_grids),
        )
        compute_ccv_policy_functions.append(compute_ccv_argmax)

        # create list of emax_calculators
        # ==============================================================================
        calculator = get_solve_discrete_problem(
            random_utility_shock_type=_mod.random_utility_shocks,
            variable_info=_mod.variable_info,
            is_last_period=is_last_period,
        )
        emax_calculators.append(calculator)

    # ==================================================================================
    # select requested solver and partial arguments into it
    # ==================================================================================
    _solve_model = partial(
        solve,
        state_choice_spaces=state_choice_spaces,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_functions=compute_ccv_functions,
        emax_calculators=emax_calculators,
        logger=logger,
    )

    solve_model = jax.jit(_solve_model) if jit else _solve_model

    _next_state_simulate = get_next_state_function(model=_mod, target="simulate")
    simulate_model = partial(
        simulate,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_policy_functions=compute_ccv_policy_functions,
        model=_mod,
        next_state=jax.jit(_next_state_simulate),
        logger=logger,
    )

    if targets == "solve":
        _target = solve_model
    elif targets == "simulate":
        _target = simulate_model
    elif targets == "solve_and_simulate":
        _target = partial(simulate_model, solve_model=solve_model)

    return cast(Callable, _target), _mod.params


def create_compute_conditional_continuation_value(
    utility_and_feasibility,
    continuous_choice_variables,
):
    """Create a function that computes the conditional continuation value.

    Note:
    -----
    This function solves the continuous choice problem conditional on a state-
    (discrete-)choice combination.

    Args:
        utility_and_feasibility (callable): A function that takes a state-choice
            combination and return the utility of that combination (float) and whether
            the state-choice combination is feasible (bool).
        continuous_choice_variables (list): List of choice variable names that are
            continuous.

    Returns:
        callable: A function that takes a state-choice combination and returns the
            conditional continuation value over the continuous choices.

    """
    if continuous_choice_variables:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_choice_variables,
        )

    @functools.wraps(utility_and_feasibility)
    def compute_ccv(*args, **kwargs):
        u, f = utility_and_feasibility(*args, **kwargs)
        return u.max(where=f, initial=-jnp.inf)

    return compute_ccv


def create_compute_conditional_continuation_policy(
    utility_and_feasibility,
    continuous_choice_variables,
):
    """Create a function that computes the conditional continuation policy.

    Note:
    -----
    This function solves the continuous choice problem conditional on a state-
    (discrete-)choice combination.

    Args:
        utility_and_feasibility (callable): A function that takes a state-choice
            combination and return the utility of that combination (float) and whether
            the state-choice combination is feasible (bool).
        continuous_choice_variables (list): List of choice variable names that are
            continuous.

    Returns:
        callable: A function that takes a state-choice combination and returns the
            conditional continuation value over the continuous choices, and the index
            that maximizes the conditional continuation value.

    """
    if continuous_choice_variables:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_choice_variables,
        )

    @functools.wraps(utility_and_feasibility)
    def compute_ccv_policy(*args, **kwargs):
        u, f = utility_and_feasibility(*args, **kwargs)
        _argmax, _max = argmax(u, where=f, initial=-jnp.inf)
        return _argmax, _max

    return compute_ccv_policy
