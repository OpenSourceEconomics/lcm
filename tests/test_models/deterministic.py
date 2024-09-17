"""Example specifications of a deterministic consumption-saving model.

The specification builds on the example model presented in the paper: "The endogenous
grid method for discrete-continuous dynamic choice models with (or without) taste
shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643).

"""

from copy import deepcopy
from dataclasses import dataclass

import jax.numpy as jnp

from lcm import DiscreteGrid, LinspaceGrid, Model

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
# Dataclasses can be used to represent labeled versions of discrete grids. For this, you
# need to define a dataclass with one field per label with the same value as used in the
# grid definition. They are especially useful to make case distinctions more readable.
# One example can be found in the absorbing_retirement_filter function below.
# --------------------------------------------------------------------------------------
@dataclass
class RetirementStatus:
    working: int = 0
    retired: int = 1


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
    return utility(consumption, working, disutility_of_work)


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


# For discrete state variables, we need to assure that the next state also belongs to
# the grid. We therefore round the result of the continuous state transition function.
def next_wealth_discrete(wealth, consumption, labor_income, interest_rate):
    next_wealth_cont = next_wealth(wealth, consumption, labor_income, interest_rate)
    return jnp.rint(next_wealth_cont).astype(jnp.int32)


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth):
    return consumption <= wealth


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
def absorbing_retirement_filter(retirement, lagged_retirement):
    return jnp.logical_or(
        retirement == RetirementStatus.retired,
        lagged_retirement == RetirementStatus.working,
    )


# ======================================================================================
# Model specifications
# ======================================================================================

ISKHAKOV_ET_AL_2017 = Model(
    description=(
        "Corresponds to the example model in Iskhakov et al. (2017). In comparison to "
        "the extensions below, wage is treated as a constant parameter and therefore "
        "there is no need for the wage and age functions."
    ),
    n_periods=3,
    functions={
        "utility": utility_with_filter,
        "next_wealth": next_wealth,
        "next_lagged_retirement": lambda retirement: retirement,
        "consumption_constraint": consumption_constraint,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "labor_income": labor_income,
        "working": working,
    },
    choices={
        "retirement": DiscreteGrid([0, 1]),
        "consumption": LinspaceGrid(
            start=1,
            stop=400,
            n_points=500,
        ),
    },
    states={
        "wealth": LinspaceGrid(
            start=1,
            stop=400,
            n_points=100,
        ),
        "lagged_retirement": DiscreteGrid([0, 1]),
    },
)


ISKHAKOV_ET_AL_2017_STRIPPED_DOWN = Model(
    description=(
        "Starts from Iskhakov et al. (2017), removes filters and the lagged_retirement "
        "state, and adds wage function that depends on age."
    ),
    n_periods=3,
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "working": working,
        "wage": wage,
        "age": age,
    },
    choices={
        "retirement": DiscreteGrid([0, 1]),
        "consumption": LinspaceGrid(
            start=1,
            stop=400,
            n_points=500,
        ),
    },
    states={
        "wealth": LinspaceGrid(
            start=1,
            stop=400,
            n_points=100,
        ),
    },
)


ISKHAKOV_ET_AL_2017_FULLY_DISCRETE = Model(
    description=(
        "Starts from Iskhakov et al. (2017), removes filters and the lagged_retirement "
        "state, and makes the consumption decision and the wealth state discrete."
    ),
    n_periods=3,
    functions={
        "utility": utility,
        "next_wealth": next_wealth_discrete,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "working": working,
    },
    choices={
        "retirement": DiscreteGrid([0, 1]),
        "consumption": DiscreteGrid([1, 2]),
    },
    states={
        "wealth": DiscreteGrid(list(range(1, 401))),
    },
)


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
    return model_config.replace(n_periods=n_periods)


def get_params(beta=0.95, disutility_of_work=0.25, interest_rate=0.05, wage=5.0):
    return {
        "beta": beta,
        "utility": {"disutility_of_work": disutility_of_work},
        "next_wealth": {
            "interest_rate": interest_rate,
        },
        "labor_income": {"wage": wage},
    }
