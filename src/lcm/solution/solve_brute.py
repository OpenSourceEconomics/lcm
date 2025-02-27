import logging
from collections.abc import Callable

import jax
from jax import Array

from lcm.dispatchers import productmap
from lcm.interfaces import StateChoiceSpace
from lcm.typing import DiscreteProblemValueSolverFunction, ParamsDict


def solve(
    params: ParamsDict,
    state_choice_spaces: dict[int, StateChoiceSpace],
    compute_ccv_functions: dict[int, Callable[[Array, Array], Array]],
    emax_calculators: dict[int, DiscreteProblemValueSolverFunction],
    logger: logging.Logger,
) -> dict[int, Array]:
    """Solve a model by brute force.

    Notes:
    -----
    - For now, do not do any caching. Later lists with large objects can be replaced
      by file paths to cached versions.
    - For simplicity, we always have lists of length n_periods with state_spaces, ...
      even if in the model the those things don't vary across periods. As long as
      we don't make copies, this has no memory overhead, but it simplifies the backwards
      loop.

    Args:
        params: Dict of model parameters.
        state_choice_spaces: Dict with one state_choice_space per period.
        compute_ccv_functions: Dict with one function needed to solve the agent's
            problem. Each function depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - params
        emax_calculators: List of functions that take continuation
            values for combinations of states and discrete choices and calculate the
            expected maximum over all discrete choices of a given state.
        logger: Logger that logs to stdout.

    Returns:
        Dict with one value function array per period.

    """
    n_periods = len(state_choice_spaces)
    solution = {}
    vf_arr = None

    logger.info("Starting solution")

    # backwards induction loop
    for period in reversed(range(n_periods)):
        # solve continuous problem, conditional on discrete choices
        conditional_continuation_values = solve_continuous_problem(
            state_choice_space=state_choice_spaces[period],
            compute_ccv=compute_ccv_functions[period],
            vf_arr=vf_arr,
            params=params,
        )

        # solve discrete problem by calculating expected maximum over discrete choices
        calculate_emax = emax_calculators[period]
        vf_arr = calculate_emax(conditional_continuation_values, params=params)
        solution[period] = vf_arr

        logger.info("Period: %s", period)

    return solution


def solve_continuous_problem(
    state_choice_space: StateChoiceSpace,
    compute_ccv: Callable[..., Array],
    vf_arr: Array | None,
    params: ParamsDict,
) -> Array:
    """Solve the agent's continuous choices problem problem.

    Args:
        state_choice_space: Class with model variables.
        compute_ccv: Function that returns the conditional continuation
            values for a given combination of states and discrete choices. The function
            depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - params
        vf_arr: Value function array.
        params: Dict of model parameters.

    Returns:
        Jax array with continuation values for each combination of the set of states and
            the set of discrete choices. The number and order of dimensions is defined
            by the `gridmap` function.

    """
    _gridmapped = productmap(
        func=compute_ccv,
        variables=state_choice_space.ordered_var_names,
    )
    gridmapped = jax.jit(_gridmapped)

    return gridmapped(
        **state_choice_space.states,
        **state_choice_space.discrete_choices,
        **state_choice_space.continuous_choices,
        vf_arr=vf_arr,
        params=params,
    )
