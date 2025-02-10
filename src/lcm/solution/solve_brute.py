import jax

from lcm.dispatchers import spacemap
from lcm.interfaces import SolutionSpace


def solve(
    params,
    state_choice_spaces,
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
        continuous_choice_grids (list): List of dicts with 1d grids for continuous
            choice variables.
        compute_ccv_functions (list): List of functions needed to solve the agent's
            problem. Each function depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
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
            params=params,
        )

        # solve discrete problem by calculating expected maximum over discrete choices
        calculate_emax = emax_calculators[period]
        vf_arr = calculate_emax(conditional_continuation_values, params=params)
        reversed_solution.append(vf_arr)

        logger.info("Period: %s", period)

    return list(reversed(reversed_solution))


def solve_continuous_problem(
    state_choice_space: SolutionSpace,
    compute_ccv,
    continuous_choice_grids,
    vf_arr,
    params,
):
    """Solve the agent's continuous choices problem problem.

    Args:
        state_choice_space (Space): Class with model variables.
        compute_ccv (callable): Function that returns the conditional continuation
            values for a given combination of states and discrete choices. The function
            depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - params
        continuous_choice_grids (list): List of dicts with 1d grids for continuous
            choice variables.
        vf_arr (jax.numpy.ndarray): Value function array.
        params (dict): Dict of model parameters.

    Returns:
        jnp.ndarray: Jax array with continuation values for each combination of a
            state and a discrete choice. The number and order of dimensions is defined
            by the `gridmap` function.

    """
    _gridmapped = spacemap(
        func=compute_ccv,
        dense_vars=list(state_choice_space.vars),
    )
    gridmapped = jax.jit(_gridmapped)

    return gridmapped(
        **state_choice_space.vars,
        **continuous_choice_grids,
        vf_arr=vf_arr,
        params=params,
    )
