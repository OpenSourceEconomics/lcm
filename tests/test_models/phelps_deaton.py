"""Example specifications of the Phelps-Deaton model."""

import jax.numpy as jnp

# ======================================================================================
# Numerical parameters and constants
# ======================================================================================

N_GRID_POINTS = {
    "states": 100,
    "choices": 500,
}

RETIREMENT_AGE = 65

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility(consumption, working, delta):
    return jnp.log(consumption) - delta * working


def utility_with_filter(
    consumption,
    working,
    delta,
    lagged_retirement,  # noqa: ARG001
):
    return utility(consumption=consumption, working=working, delta=delta)


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def working(retirement):
    return 1 - retirement


def wage(age):
    return 1 + 0.1 * age


def age(_period):
    return _period + 18


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + wage * working


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth):
    return consumption <= wealth


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
def mandatory_retirement_filter(retirement, age):
    return jnp.logical_or(retirement == 1, age < RETIREMENT_AGE)


def absorbing_retirement_filter(retirement, lagged_retirement):
    return jnp.logical_or(retirement == 1, lagged_retirement == 0)


# ======================================================================================
# Model specification and parameters
# ======================================================================================

PHELPS_DEATON = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
        "wage": wage,
        "age": age,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_GRID_POINTS["choices"],
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_GRID_POINTS["states"],
        },
    },
    "n_periods": 3,
}


PHELPS_DEATON_FULLY_DISCRETE = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {"options": [1, 2]},
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_GRID_POINTS["states"],
        },
    },
    "n_periods": 3,
}


PHELPS_DEATON_WITH_FILTERS = {
    "functions": {
        "utility": utility_with_filter,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "next_lagged_retirement": lambda retirement: retirement,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_GRID_POINTS["choices"],
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_GRID_POINTS["states"],
        },
        "lagged_retirement": {"options": [0, 1]},
    },
    "n_periods": 3,
}