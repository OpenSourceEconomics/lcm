from functools import partial

import jax.numpy as jnp
from lcm.dispatchers import spacemap


def solve(
    params,
    state_choice_spaces,
    value_function_evaluators,
    state_indexers,
    continuous_choice_grids,
    agent_functions,
    emax_calculators,
    choice_segments,
):
    """Solve a model by brute force.

    Notes
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
        agent_functions (list): List of functions needed to solve the agents problem.
            The functions take state variables, choice variables and params as arguments
            and return three things:
            - float: The achieved utility
            - bool: Whether the state choice combination is feasible
            - dict: The next values of the state variables.
        emax_calculators (list): List of functions that take continuation
            values for combinations of states and discrete choices and calculate the
            expected maximum over all discrete choices of a given state.
        choice_segments (list): List of arrays or None with the choice segments that
            indicate which sparse choice variables belong to one state.

    """
    # ==================================================================================
    # extract information
    # ==================================================================================
    n_periods = len(state_choice_spaces)
    last_period = n_periods - 1

    # ==================================================================================
    # initialize result lists
    # ==================================================================================

    vf_arrs = []
    vf_arr = None

    # ==================================================================================
    # get last period solution
    # ==================================================================================

    # calculate continuation values conditional on a discrete choice
    conditional_continuation_values = solve_continuous_problem(
        state_choice_space=state_choice_spaces[last_period],
        utility_and_feasibility=agent_functions[last_period],
        continuous_choice_grids=continuous_choice_grids[last_period],
        vf_arr=vf_arr,
        params=params,
    )

    # calculate the expected maximum over the discrete choices in each state
    calculate_emax = emax_calculators[last_period]
    vf_arr = calculate_emax(
        values=conditional_continuation_values,
        choice_segments=choice_segments,
        params=params,
    )

    # append the vf_arr to results and wrap it into a function evaluator for next step
    vf_arrs.append(vf_arr)
    value_function = partial(  # noqa: F841
        value_function_evaluators[last_period],
        vf_arr=vf_arr,
        **state_indexers[last_period],
    )

    # ==================================================================================
    # backwards induction loop
    # ==================================================================================

    # contsolve generic period
    # emax aggregation

    return vf_arrs


def solve_continuous_problem(
    state_choice_space,
    utility_and_feasibility,
    continuous_choice_grids,
    vf_arr,
    params,
):
    """Solve the agent's continuous choices problem problem.

    Args:
        state_choice_space
        utility_and_feasibility (callable): Function that returns a tuple where the
            first entry is the utility and the second is a bool that indicates
            feasibility. The function depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - params
        continuous_choice_grids (dict)
        vf_arr (jax.numpy.ndarray)
        params (dict)

    Returns:
        np.ndarray: Numpy array with continuation values for each combination of a
            state and a discrete choice. The number and order of dimensions is defined
            by the ``gridmap`` function.

    """
    # ==================================================================================
    # extract information
    # ==================================================================================
    n_sparse = len(state_choice_space.sparse_vars)
    n_dense = len(state_choice_space.dense_vars)
    n_cont_choices = len(continuous_choice_grids)

    # ==================================================================================
    # find axes over which we want to take the maximum
    # ==================================================================================
    offset = n_sparse + n_dense
    max_axes = tuple(range(offset, offset + n_cont_choices))

    # ==================================================================================
    # apply dispatcher
    # ==================================================================================
    gridmapped = spacemap(
        func=utility_and_feasibility,
        dense_vars=list(state_choice_space.dense_vars) + list(continuous_choice_grids),
        sparse_vars=list(state_choice_space.sparse_vars),
        dense_first=False,
    )

    utilities, feasibilities = gridmapped(
        **state_choice_space.dense_vars,
        **continuous_choice_grids,
        **state_choice_space.sparse_vars,
        vf_arr=vf_arr,
        params=params,
    )

    best = utilities.max(axis=max_axes, where=feasibilities, initial=-jnp.inf)

    return best
