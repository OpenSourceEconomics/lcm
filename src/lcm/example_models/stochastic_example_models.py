"""Define example model specifications."""

import jax.numpy as jnp

import lcm

N_CHOICE_GRID_POINTS = 500
N_STATE_GRID_POINTS = 100


def utility(consumption, working, health, partner, delta, gamma):  # noqa: ARG001
    return jnp.log(consumption) + (gamma * health - delta) * working


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + wage * working


@lcm.mark.stochastic
def next_health(health, partner):  # noqa: ARG001
    pass


@lcm.mark.stochastic
def next_partner(_period, working, partner):  # noqa: ARG001
    pass


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


MODEL = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "next_partner": next_partner,
        "consumption_constraint": consumption_constraint,
    },
    "choices": {
        "working": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_CHOICE_GRID_POINTS,
        },
    },
    "states": {
        "health": {"options": [0, 1]},
        "partner": {"options": [0, 1]},
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
    },
    "n_periods": 3,
}


PARAMS = {
    "beta": 0.95,
    "utility": {"delta": 0.5, "gamma": 0.25},
    "next_wealth": {"interest_rate": 0.05, "wage": 10.0},
    "next_health": {},
    "consumption_constraint": {},
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
