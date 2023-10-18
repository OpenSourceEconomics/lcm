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
def next_partner(_period):
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
            "start": 0,
            "stop": 100,
            "n_points": N_CHOICE_GRID_POINTS,
        },
    },
    "states": {
        "health": {"options": [0, 1]},
        "partner": {"options": [0, 1]},
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
    },
    "n_periods": 3,
}


PARAMS = {
    "beta": 0.25,
    "utility": {"delta": 0.25, "gamma": 0.25},
    "next_wealth": {"interest_rate": 0.25, "wage": 0.25},
    "next_health": {},
    "consumption_constraint": {},
    "shocks": {
        "health": jnp.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]),
        "partner": jnp.array([[1.0, 0], [0.0, 1]]),
    },
}
