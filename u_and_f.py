


# ======================================================================================
# Non stochastic
# ======================================================================================
def utility_and_feasibility(health, wealth, consumption, working, params, vf_arr): # helpers like indexers, ...
    # evaluate next functions
    next_w = next_wealth(
        wealth,
        consumption,
        working,
        params["next_wealth"]["wage"],
        params["next_wealth"]["interest_rate"],
    )

    next_h = health


    # ----------------------------------------------------------------------------------
    # function_evaluator
    # ----------------------------------------------------------------------------------
    # vf_arr has dimensions (health, wealth) and shape (2, n_wealth_points)
    # discrete lookup
    continuous_part = vf_arr[next_h]
    # continuous interpolation
    next_w_pos = np.searchsorted(wealth_grid, next_w)
    ccv = interpolate(continuous_part, next_w_pos)
    # ----------------------------------------------------------------------------------

    # compute big_u
    u = utility(consumption, working, health, params["utility"]["delta"])
    big_u = u + params["beta"] * ccv

    # compute feasibility
    feasible = consumption <= wealth
    return big_u, feasible

# ======================================================================================
# Stochastic
# ======================================================================================
def utility_and_feasibility(health, wealth, consumption, working, params, vf_arr): # helpers like indexers, ...
    next_w = next_wealth(
        wealth,
        consumption,
        working,
        params["next_wealth"]["wage"],
        params["next_wealth"]["interest_rate"],
    )

    # get the health grid and weights
    next_h_points, next_health_weights = __next_health__(health, params)

    # could be an outer product over multiple independent shock dimensions
    outcome_weights = next_health_weights

    # vmap the scalar function evaluator over the stochastic state dimensions
    vmapped_function_evaluator = jax.vmap(
        __scalar_function_evaluator__,
        in_axes=(0, None, None),
    )

    # evaluate the vmapped function
    ccv_outcomes = vmapped_function_evaluator(
        next_h_points,
        next_w,
        vf_arr,
    )

    # reduce over outcomes
    ccv = (ccv_outcomes * outcome_weights).sum()

    u = utility(consumption, working, health, params["utility"]["delta"])
    big_u = u + params["beta"] * ccv

    feasible = consumption <= wealth
    return big_u, feasible

# Will be dynamically generated from the decorated function
def __next_health__(health, params, health_grid):
    health_pos = get_pos(health, health_grid)
    health_weights = params["shocks"]["health"][health_pos]
    return health_grid, health_weights


# Will be dynamically generated
def __scalar_function_evaluator__(health, wealth, vf_arr):
    # vf_arr has dimensions (health, wealth) and shape (2, n_wealth_points)
    # discrete lookup
    continuous_part = vf_arr[next_h]
    # continuous interpolation
    next_w_pos = np.searchsorted(wealth_grid, next_w)
    ccv = interpolate(continuous_part, next_w_pos)
    return ccv

@with_signature([*STATE_VARIABLES, *CHOICE_VARIABLES, "params", "vf_arr"])
def dream_utility_and_feasibility(**kwargs):
    states = {k: v for k, v in kwargs.items() if k in STATE_VARIABLES}
    choices = {k: v for k, v in kwargs.items() if k in CHOICE_VARIABLES}
    # next_states = dict of floats and grids for all state variables
    # weights = dict of arrays for stochastic state variables
    next_states, weights = __next_states__(**states, **choices, params)

    # outcome_weights are an array with weights for all possible outcomes;
    # outcomes are the cartesian product of all possible outcomes of all stochastic
    # state variables
    outcome_weights = aggregate_weights(weights)

    function_evaluator = productmap(__scalar_function_evaluator__, *weights)

    # evaluate the vmapped function
    ccv_outcomes = function_evaluator(
        **next_states,
        vf_arr,
    )

    # reduce over outcomes
    ccv = (ccv_outcomes * outcome_weights).sum()

    u, f = __small_u_and_f__(**states, **choices, params)

    big_u = u + params["beta"] * ccv

    return big_u, f





# ======================================================================================
# Questions
# ======================================================================================

# Implement the `mark.stochastic` decorator
# Write get_next_function
# Write aggregate_independent_probabilities
# Implement check that no constraint depends on stochastic next functions
# Implement new get_utility_and_feasibility