"""Define example model specifications."""

import jax.numpy as jnp

import lcm

N_CHOICE_GRID_POINTS = 500
N_STATE_GRID_POINTS = 100


def phelps_deaton_utility(consumption, working, delta):
    return jnp.log(consumption) - delta * working


def working(retirement):
    return 1 - retirement


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + wage * working


@lcm.mark.stochastic
def next_wage(age, wage):  # noqa: ARG001
    pass


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


def age(period):
    return period + 18


PHELPS_DEATON = {
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "next_wage": next_wage,
        "consumption_constraint": consumption_constraint,
        "working": working,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_CHOICE_GRID_POINTS,
        },
    },
    "states": {
        "wage": {"options": [0, 1]},
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
    },
    "n_periods": 20,
}
