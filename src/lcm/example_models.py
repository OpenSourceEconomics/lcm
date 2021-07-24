"""Define example model specifications."""
import numpy as np


def phelps_deaton_utility_with_shocks(
    consumption, working, delta, wage_shock, additive_utility_shock
):
    return np.log(consumption) + additive_utility_shock - delta * wage_shock * working


def phelps_deaton_utility(consumption, working, delta, wage_shock):
    return np.log(consumption) - delta * wage_shock * working


def working(retirement):
    return 1 - retirement


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return interest_rate * (wealth - consumption) + wage * working


def next_wealth_constraint(next_wealth):
    return next_wealth >= 0


PHELPS_DEATON_WITH_SHOCKS = {
    "functions": {
        "utility": phelps_deaton_utility_with_shocks,
        "next_wealth": next_wealth,
        "next_wealth_constraint": next_wealth_constraint,
        "working": working,
    },
    "choices": {
        "retirement": {"options": [0, 1], "absorbing_values": [1]},
        "consumption": {"grid_type": "linspace", "n_points": 10},
    },
    "states": {"wealth": {"grid_type": "linspace", "n_points": 12}},
    "shocks": {
        "wage_shock": "lognormal",
        # special name to signal that this shock can be set to zero to calculate
        # expected utility
        "additive_utility_shock": "extreme_value",
    },
}


PHELPS_DEATON = {
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "next_wealth_constraint": next_wealth_constraint,
        "working": working,
    },
    "choices": {
        "retirement": {"options": [0, 1], "absorbing_values": [1]},
        "consumption": {"grid_type": "linspace", "n_points": 10},
    },
    "states": {"wealth": {"grid_type": "linspace", "n_points": 12}},
}
