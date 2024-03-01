"""Example specifications of a deterministic consumption-saving model.

The specification builds on the example model presented in the paper: "The endogenous
grid method for discrete-continuous dynamic choice models with (or without) taste
shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643).

"""

from copy import deepcopy

import jax.numpy as jnp

# ======================================================================================
# Numerical parameters and constants
# ======================================================================================

N_GRID_POINTS = {
    "wealth": 100,
    "consumption": 500,
}

RETIREMENT_AGE = 65

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility(consumption, working, disutility_of_work):
    return jnp.log(consumption) - disutility_of_work * working


def utility_with_filter(
    consumption,
    working,
    disutility_of_work,
    # Temporary workaround for bug described in issue #30, which requires us to pass
    # all state variables to the utility function.
    # TODO(@timmens): Remove function once #30 is fixed (re-use "utility").
    # https://github.com/OpenSourceEconomics/lcm/issues/30
    lagged_retirement,  # noqa: ARG001
):
    return utility(consumption, working=working, disutility_of_work=disutility_of_work)


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(working, wage):
    return working * wage


def working(retirement):
    return 1 - retirement


def wage(age):
    return 1 + 0.1 * age


def age(_period):
    return _period + 18


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, labor_income, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + labor_income


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
# Model specifications
# ======================================================================================

BASE_MODEL = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "working": working,
        "wage": wage,
        "age": age,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": N_GRID_POINTS["consumption"],
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": N_GRID_POINTS["wealth"],
        },
    },
    "n_periods": 3,
}


BASE_MODEL_FULLY_DISCRETE = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "working": working,
        "wage": wage,
        "age": age,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {"options": [1, 2]},
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": N_GRID_POINTS["wealth"],
        },
    },
    "n_periods": 3,
}


BASE_MODEL_WITH_FILTERS = {
    "functions": {
        "utility": utility_with_filter,
        "next_wealth": next_wealth,
        "next_lagged_retirement": lambda retirement: retirement,
        "consumption_constraint": consumption_constraint,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "labor_income": labor_income,
        "working": working,
        "wage": wage,
        "age": age,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": N_GRID_POINTS["consumption"],
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": N_GRID_POINTS["wealth"],
        },
        "lagged_retirement": {"options": [0, 1]},
    },
    "n_periods": 3,
}


ISKHAKOV_ET_AL_2017 = {
    "functions": {
        "utility": utility_with_filter,
        "next_wealth": next_wealth,
        "next_lagged_retirement": lambda retirement: retirement,
        "consumption_constraint": consumption_constraint,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "labor_income": labor_income,
        "working": working,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": N_GRID_POINTS["consumption"],
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": N_GRID_POINTS["wealth"],
        },
        "lagged_retirement": {"options": [0, 1]},
    },
    "n_periods": 3,
}


# ======================================================================================
# Get models and params
# ======================================================================================

IMPLEMENTED_MODELS = {
    "base": BASE_MODEL,
    "fully_discrete": BASE_MODEL_FULLY_DISCRETE,
    "with_filters": BASE_MODEL_WITH_FILTERS,
    "iskhakov_et_al_2017": ISKHAKOV_ET_AL_2017,
}


def get_model_config(model_name: str, n_periods: int):
    model_config = deepcopy(IMPLEMENTED_MODELS[model_name])
    model_config["n_periods"] = n_periods
    return model_config


def get_params(beta=0.95, disutility_of_work=0.25, interest_rate=0.05, wage=5.0):
    return {
        "beta": beta,
        "utility": {"disutility_of_work": disutility_of_work},
        "next_wealth": {
            "interest_rate": interest_rate,
        },
        "labor_income": {"wage": wage},
    }
