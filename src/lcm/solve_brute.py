from functools import partial

import jax

from lcm.dispatchers import spacemap


def solve(
    params,
    state_choice_spaces,
    state_indexers,
    continuous_choice_grids,
    compute_ccv_functions,
    emax_calculators,
    logger,
):
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
        params (dict): Dict of model parameters.
        state_choice_spaces (list): List with one state_choice_space per period.
        value_function_evaluators (list): List with one value_function_evaluator per
            period.
        state_indexers (list): List of dicts with length n_periods. Each dict contains
            one or several state indexers.
        continuous_choice_grids (list): List of dicts with 1d grids for continuous
            choice variables.
        compute_ccv_functions (list): List of functions needed to solve the agent's
            problem. Each function depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - one or several state_indexers
            - params
        emax_calculators (list): List of functions that take continuation
            values for combinations of states and discrete choices and calculate the
            expected maximum over all discrete choices of a given state.
        logger (logging.Logger): Logger that logs to stdout.

    Returns:
        list: List with one value function array per period.

    """
    # extract information
    n_periods = len(state_choice_spaces)
    reversed_solution = []
    vf_arr = None

    logger.info("Starting solution")

    # backwards induction loop
    for period in reversed(range(n_periods)):
        # solve continuous problem, conditional on discrete choices
        conditional_continuation_values = solve_continuous_problem(
            state_choice_space=state_choice_spaces[period],
            compute_ccv=compute_ccv_functions[period],
            continuous_choice_grids=continuous_choice_grids[period],
            vf_arr=vf_arr,
            state_indexers=state_indexers[period],
            params=params,
        )

        # solve discrete problem by calculating expected maximum over discrete choices
        calculate_emax = emax_calculators[period]
        vf_arr = calculate_emax(conditional_continuation_values)
        reversed_solution.append(vf_arr)

        logger.info("Period: %s", period)

    return list(reversed(reversed_solution))


@partial(jax.jit, static_argnums=1)
def solve_continuous_problem(
    state_choice_space,
    compute_ccv,
    continuous_choice_grids,
    vf_arr,
    state_indexers,
    params,
):
    """Solve the agent's continuous choices problem problem.

    Args:
        state_choice_space (Space): Namedtuple with entries dense_vars and sparse_vars.
        compute_ccv (callable): Function that returns the conditional continuation
            values for a given combination of states and discrete choices. The function
            depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - one or several state_indexers
            - params
        continuous_choice_grids (list): List of dicts with 1d grids for continuous
            choice variables.
        vf_arr (jax.numpy.ndarray): Value function array.
        state_indexers (list): List of dicts with length n_periods. Each dict contains
            one or several state indexers.
        params (dict): Dict of model parameters.

    Returns:
        jnp.ndarray: Jax array with continuation values for each combination of a
            state and a discrete choice. The number and order of dimensions is defined
            by the ``gridmap`` function.

    """
    gridmapped = spacemap(
        func=compute_ccv,
        dense_vars=list(state_choice_space.dense_vars),
        sparse_vars=list(state_choice_space.sparse_vars),
        dense_first=False,
    )

    return gridmapped(
        **state_choice_space.dense_vars,
        **continuous_choice_grids,
        **state_choice_space.sparse_vars,
        **state_indexers,
        vf_arr=vf_arr,
        params=params,
    )
