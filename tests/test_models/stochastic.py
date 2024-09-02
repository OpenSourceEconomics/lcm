"""Example specification of a stochastic consumption-saving model.

This specification is motivated by the example model presented in the paper: "The
endogenous grid method for discrete-continuous dynamic choice models with (or without)
taste shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning
(2017, https://doi.org/10.3982/QE643).

See also the specifications in tests/test_models/deterministic.py.

"""

from copy import deepcopy

import jax.numpy as jnp

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Model

# ======================================================================================
# Numerical parameters and constants
# ======================================================================================

N_GRID_POINTS = {
    "wealth": 100,
    "consumption": 200,
}

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    consumption,
    working,
    health,
    # Temporary workaround for bug described in issue #30, which requires us to pass
    # all state variables to the utility function.
    # TODO(@timmens): Remove function arguments once #30 is fixed.
    # https://github.com/OpenSourceEconomics/lcm/issues/30
    partner,  # noqa: ARG001
    disutility_of_work,
):
    return jnp.log(consumption) - (1 - health / 2) * disutility_of_work * working


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(working, wage):
    return working * wage


# --------------------------------------------------------------------------------------
# Deterministic state transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, labor_income, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + labor_income


# --------------------------------------------------------------------------------------
# Stochastic state transitions
# --------------------------------------------------------------------------------------
@lcm.mark.stochastic
def next_health(health, partner):
    pass


@lcm.mark.stochastic
def next_partner(_period, working, partner):
    pass


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth):
    return consumption <= wealth


# ======================================================================================
# Model specification and parameters
# ======================================================================================

MODEL_CONFIG = Model(
    description=(
        "Starts from Iskhakov et al. (2017), removes filters and the lagged_retirement "
        "state, and adds discrete stochastic state variables health and partner."
    ),
    n_periods=3,
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "next_partner": next_partner,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
    },
    choices={
        "working": DiscreteGrid([0, 1]),
        "consumption": LinspaceGrid(
            start=1,
            stop=100,
            n_points=N_GRID_POINTS["consumption"],
        ),
    },
    states={
        "health": DiscreteGrid([0, 1]),
        "partner": DiscreteGrid([0, 1]),
        "wealth": LinspaceGrid(
            start=1,
            stop=100,
            n_points=N_GRID_POINTS["wealth"],
        ),
    },
)


# ======================================================================================
# Get models and params
# ======================================================================================

IMPLEMENTED_MODELS = {
    "only_discrete_vars_stochastic": MODEL_CONFIG,
}


def get_model_config(model_name: str, n_periods: int):
    model_config = deepcopy(IMPLEMENTED_MODELS[model_name])
    return model_config.replace(n_periods=n_periods)


def get_params(
    beta=0.95,
    disutility_of_work=0.5,
    interest_rate=0.05,
    wage=10.0,
    health_transition=None,
    partner_transition=None,
):
    # ----------------------------------------------------------------------------------
    # Transition matrices
    # ----------------------------------------------------------------------------------

    # Health shock transition:
    # ------------------------------------------------------------------------------
    # 1st dimension: Current health state
    # 2nd dimension: Current Partner state
    # 3rd dimension: Probability distribution over next period's health state
    default_health_transition = jnp.array(
        [
            # Current health state 0
            [
                # Current Partner state 0
                [0.9, 0.1],
                # Current Partner state 1
                [0.5, 0.5],
            ],
            # Current health state 1
            [
                # Current Partner state 0
                [0.5, 0.5],
                # Current Partner state 1
                [0.1, 0.9],
            ],
        ],
    )
    health_transition = (
        default_health_transition if health_transition is None else health_transition
    )

    # Partner shock transition:
    # ------------------------------------------------------------------------------
    # 1st dimension: The period
    # 2nd dimension: Current working decision
    # 3rd dimension: Current partner state
    # 4th dimension: Probability distribution over next period's partner state
    default_partner_transition = jnp.array(
        [
            # Transition from period 0 to period 1
            [
                # Current working decision 0
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [1.0, 0],
                ],
                # Current working decision 1
                [
                    # Current partner state 0
                    [0, 1.0],
                    # Current partner state 1
                    [0.0, 1.0],
                ],
            ],
            # Transition from period 1 to period 2
            [
                # Description is the same as above
                [[0, 1.0], [1.0, 0]],
                [[0, 1.0], [0.0, 1.0]],
            ],
        ],
    )
    partner_transition = (
        default_partner_transition if partner_transition is None else partner_transition
    )

    # ----------------------------------------------------------------------------------
    # Model parameters
    # ----------------------------------------------------------------------------------
    return {
        "beta": beta,
        "utility": {"disutility_of_work": disutility_of_work},
        "next_wealth": {"interest_rate": interest_rate},
        "next_health": {},
        "consumption_constraint": {},
        "labor_income": {"wage": wage},
        "shocks": {
            "health": health_transition,
            "partner": partner_transition,
        },
    }
