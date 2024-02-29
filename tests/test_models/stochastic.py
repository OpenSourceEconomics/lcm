"""Example specifications of a stochastic consumption-saving model.

This specification is motivated by the example model presented in the paper: "The
endogenous grid method for discrete-continuous dynamic choice models with (or without)
taste shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning
(2017, https://doi.org/10.3982/QE643).

See also the specifications in tests/test_models/deterministic.py.

"""

import jax.numpy as jnp
import lcm

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
    # TODO(@timmens): Remove function arguments once #30 is fixed (re-use "utility").
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
def next_health(health, partner):  # noqa: ARG001
    pass


@lcm.mark.stochastic
def next_partner(_period, working, partner):  # noqa: ARG001
    pass


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
        "next_health": next_health,
        "next_partner": next_partner,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
    },
    "choices": {
        "working": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_GRID_POINTS["consumption"],
        },
    },
    "states": {
        "health": {"options": [0, 1]},
        "partner": {"options": [0, 1]},
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_GRID_POINTS["wealth"],
        },
    },
    "n_periods": 3,
}


PARAMS = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 0.5},
    "next_wealth": {"interest_rate": 0.05},
    "next_health": {},
    "consumption_constraint": {},
    "labor_income": {"wage": 10.0},
    "shocks": {
        # Health shock:
        # ------------------------------------------------------------------------------
        # 1st dimension: Current health state
        # 2nd dimension: Current Partner state
        # 3rd dimension: Probability distribution over next period's health state
        "health": jnp.array(
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
        ),
        # Partner shock:
        # ------------------------------------------------------------------------------
        # 1st dimension: The period
        # 2nd dimension: Current working decision
        # 3rd dimension: Current partner state
        # 4th dimension: Probability distribution over next period's partner state
        "partner": jnp.array(
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
        ),
    },
}
