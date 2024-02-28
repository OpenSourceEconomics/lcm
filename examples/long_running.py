"""Example specification for a consumption-savings model with health and leisure."""

import jax.numpy as jnp

# ======================================================================================
# Numerical parameters and constants
# ======================================================================================
N_GRID_POINTS = {
    "wealth": 100,
    "health": 100,
    "consumption": 100,
    "exercise": 200,
}

RETIREMENT_AGE = 65

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(consumption, working, health, exercise, disutility_of_work):
    return jnp.log(consumption) - (disutility_of_work - health) * working - exercise


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(wage, working):
    return wage * working


def working(leisure):
    return 1 - leisure


def wage(age):
    return 1 + 0.1 * age


def age(_period):
    return _period + 18


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, labor_income, interest_rate):
    return (1 + interest_rate) * (wealth + labor_income - consumption)


def next_health(health, exercise, working):
    return health * (1 + exercise - working / 2)


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth, labor_income):
    return consumption <= wealth + labor_income


# ======================================================================================
# Model specification and parameters
# ======================================================================================

MODEL_CONFIG = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "working": working,
        "wage": wage,
        "age": age,
    },
    "choices": {
        "leisure": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_GRID_POINTS["consumption"],
        },
        "exercise": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 1,
            "n_points": N_GRID_POINTS["exercise"],
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_GRID_POINTS["wealth"],
        },
        "health": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 1,
            "n_points": N_GRID_POINTS["health"],
        },
    },
    "n_periods": RETIREMENT_AGE - 18,
}

PARAMS = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.05},
    "next_wealth": {"interest_rate": 0.05},
}
