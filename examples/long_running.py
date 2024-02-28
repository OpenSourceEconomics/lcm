"""Example specification for a consumption-savings model with health and leisure."""

import jax.numpy as jnp

# ======================================================================================
# Numerical parameters and constants
# ======================================================================================
N_GRID_POINTS = {
    "states": 100,
    "choices": 200,
}

RETIREMENT_AGE = 65

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(consumption, working, health, sport, delta):
    return jnp.log(consumption) - (delta - health) * working - sport


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def working(leisure):
    return 1 - leisure


def wage(age):
    return 1 + 0.1 * age


def age(_period):
    return _period + 18


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth + working * wage - consumption)


def next_health(health, exercise, working):
    return health * (1 + sport - working / 2)


def next_wealth_with_shock(
    wealth,
    consumption,
    working,
    wage,
    wage_shock,
    interest_rate,
):
    return interest_rate * (wealth - consumption) + wage * wage_shock * working


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth):
    return consumption <= wealth


# ======================================================================================
# Model specification and parameters
# ======================================================================================

MODEL_CONFIG = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
        "wage": wage,
        "age": age,
        "next_health": next_health,
    },
    "choices": {
        "leisure": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_GRID_POINTS["choices"],
        },
        "sport": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 1,
            "n_points": N_GRID_POINTS["choices"],
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_GRID_POINTS["states"],
        },
        "health": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 1,
            "n_points": N_GRID_POINTS["states"],
        },
    },
    "n_periods": RETIREMENT_AGE - 18,
}

PARAMS = {
    "beta": 0.95,
    "utility": {"delta": 0.05},
    "next_wealth": {"interest_rate": 0.05},
}
