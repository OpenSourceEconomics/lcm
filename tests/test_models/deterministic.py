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


def utility_fully_discrete(
    consumption,
    working,
    disutility_of_work,
    # Temporary workaround for bug described in issue #30, which requires us to pass
    # all state variables to the utility function.
    # TODO(@timmens): Remove function once #30 is fixed (re-use "utility").
    # https://github.com/OpenSourceEconomics/lcm/issues/30
    consumption_index,  # noqa: ARG001
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


# Temporary workaround until option labels are supported that do not coincide with
# the indices of the options.
# TODO(@timmens): Remove this once #82 is closed.
# https://github.com/OpenSourceEconomics/lcm/issues/82
def consumption(consumption_index):
    _consumption_values = jnp.array([1, 2])
    return _consumption_values[consumption_index]


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
def absorbing_retirement_filter(retirement, lagged_retirement):
    return jnp.logical_or(retirement == 1, lagged_retirement == 0)


# ======================================================================================
# Model specifications
# ======================================================================================

ISKHAKOV_ET_AL_2017 = {
    "description": (
        "Corresponds to the example model in Iskhakov et al. (2017). In comparison to "
        "the extensions below, wage is treated as a constant parameter and therefore "
        "there is no need for the wage and age functions."
    ),
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


ISKHAKOV_ET_AL_2017_STRIPPED_DOWN = {
    "description": (
        "Starts from Iskhakov et al. (2017), removes filters and the lagged_retirement "
        "state, and adds wage function that depends on age."
    ),
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


ISKHAKOV_ET_AL_2017_FULLY_DISCRETE = {
    "description": (
        "Starts from Iskhakov et al. (2017), removes filters and the lagged_retirement "
        "state, and makes the consumption decision discrete."
    ),
    "functions": {
        "utility": utility_fully_discrete,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "working": working,
        "consumption": consumption,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption_index": {"options": [0, 1]},
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


# ======================================================================================
# Get models and params
# ======================================================================================

IMPLEMENTED_MODELS = {
    "iskhakov_et_al_2017": ISKHAKOV_ET_AL_2017,
    "iskhakov_et_al_2017_stripped_down": ISKHAKOV_ET_AL_2017_STRIPPED_DOWN,
    "iskhakov_et_al_2017_fully_discrete": ISKHAKOV_ET_AL_2017_FULLY_DISCRETE,
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
