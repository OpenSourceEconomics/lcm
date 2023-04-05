import jax.numpy as jnp

from lcm.dispatchers import spacemap


def solve(
    params,
    state_choice_spaces,
    state_indexers,
    continuous_choice_grids,
    utility_and_feasibility_functions,
    emax_calculators,
    choice_segments,
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
        utility_and_feasibility_functions (list): List of functions needed to solve
            the agent's problem. Each function depends on:
            - discrete and continuous state variables
            - discrete and continuous choice variables
            - vf_arr
            - one or several state_indexers
            - params
        emax_calculators (list): List of functions that take continuation
            values for combinations of states and discrete choices and calculate the
            expected maximum over all discrete choices of a given state.
        choice_segments (list): List of arrays or None with the choice segments that
            indicate which sparse choice variables belong to one state.

    Returns:
        list: List with one value function array per period.

    """
    # extract information
    n_periods = len(state_choice_spaces)
    reversed_solution = []
    vf_arr = None

    # backwards induction loop
    for period in reversed(range(n_periods)):
        # solve continuous problem, conditional on discrete choices
        conditional_continuation_values = solve_continuous_problem(
            state_choice_space=state_choice_spaces[period],
            utility_and_feasibility=utility_and_feasibility_functions[period],
            continuous_choice_grids=continuous_choice_grids[period],
            vf_arr=vf_arr,
            state_indexers=state_indexers[period],
            params=params,
        )

        # solve discrete problem by calculating expected maximum over discrete choices
        calculate_emax = emax_calculators[period]
        vf_arr = calculate_emax(
            values=conditional_continuation_values,
            choice_segments=choice_segments[period],
            params=params,
        )
        reversed_solution.append(vf_arr)

    return list(reversed(reversed_solution))


def solve_continuous_problem(
    state_choice_space,
    utility_and_feasibility,
    continuous_choice_grids,
    vf_arr,
    state_indexers,
    params,
):
    """Solve the agent's continuous choices problem problem.

    Args:
        state_choice_space (Space): Namedtuple with entries dense_vars and sparse_vars.
        utility_and_feasibility (callable): Function that returns a tuple where the
            first entry is the utility and the second is a bool that indicates
            feasibility. The function depends on:
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
    offset = n_dense
    if n_sparse > 0:
        offset += 1
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
        **state_indexers,
        vf_arr=vf_arr,
        params=params,
    )

    return utilities.max(axis=max_axes, where=feasibilities, initial=-jnp.inf)
