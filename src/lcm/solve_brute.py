import numpy as np
import jax
from functools import partial

from lcm.dispatchers import spacemap
import nvtx


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
    vf_arr = jax.numpy.empty((500, 500), np.float32)
    logger.info("Starting solution")
    funcs = []
    for period in range(n_periods-1):
        funcs.append(partial(solve_continuous_problem, state_choice_space=state_choice_spaces[period],
            compute_ccv=compute_ccv_functions[period],
            continuous_choice_grids=continuous_choice_grids[period],
            state_indexers=state_indexers[period],params=params))
    period = n_periods-1
    with nvtx.annotate("first_iter", color="blue"):
        solve_continuous_problem(vf_arr, state_choice_space=state_choice_spaces[period],
                compute_ccv=compute_ccv_functions[period],
                continuous_choice_grids=continuous_choice_grids[period],
                state_indexers=state_indexers[period],params=params)
    def solve_continuous_problem_loop(vf_arr, period):
        conditional_continuation_values = jax.lax.switch(period,funcs,vf_arr)
        # solve discrete problem by calculating expected maximum over discrete choices
        vf_arr = jax.lax.switch(period,emax_calculators,conditional_continuation_values)
        return vf_arr,vf_arr
    # backwards induction loop
    _, result = jax.lax.scan(solve_continuous_problem_loop, vf_arr, jax.numpy.arange(n_periods-1), reverse=True, length= n_periods-1)
    return result

def solve_continuous_problem(
    vf_arr,
    state_choice_space,
    compute_ccv,
    continuous_choice_grids,
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
    _gridmapped = spacemap(
        func=compute_ccv,
        dense_vars=list(state_choice_space.dense_vars),
        sparse_vars=list(state_choice_space.sparse_vars),
        put_dense_first=False,
    )
    gridmapped = jax.jit(_gridmapped)

    return gridmapped(
        **state_choice_space.dense_vars,
        **continuous_choice_grids,
        **state_choice_space.sparse_vars,
        **state_indexers,
        vf_arr=vf_arr,
        params=params,
    )
