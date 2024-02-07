"""Define example model specifications."""
import jax.numpy as jnp

RETIREMENT_AGE = 65
N_CHOICE_GRID_POINTS = 200
N_STATE_GRID_POINTS = 100


def phelps_deaton_utility(consumption, working, health, sport, delta):
    return jnp.log(consumption + 1) - (delta - health) * working - sport


def working(retirement):
    return 1 - retirement


def next_wealth_with_shock(
    wealth,
    consumption,
    working,
    wage,
    wage_shock,
    interest_rate,
):
    return interest_rate * (wealth - consumption) + wage * wage_shock * working


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + wage * working


def next_health(health, sport, working):
    return health * (1 + sport - working / 2)


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


def wage(age):
    return 1 + 0.1 * age


def age(_period):
    return _period + 18


PHELPS_DEATON = {
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
        "wage": wage,
        "age": age,
        "next_health": next_health,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_CHOICE_GRID_POINTS,
        },
        "sport": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 1,
            "n_points": N_CHOICE_GRID_POINTS,
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
        "health": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 1,
            "n_points": N_STATE_GRID_POINTS,
        },
    },
    "n_periods": RETIREMENT_AGE - 18,
}

PARAMS = {
    "beta": 0.95,
    "utility": {"delta": 0.05},
    "next_wealth": {"interest_rate": 0.05},
}
