"""Define example model specifications."""
import jax.numpy as jnp
import numpy as np

import lcm

RETIREMENT_AGE = 65
N_CHOICE_GRID_POINTS = 500
N_STATE_GRID_POINTS = 100

intensive_margin_working_hours_cats = [0, 20, 40]


def phelps_deaton_utility_with_shock(
    consumption,
    working,
    delta,
    additive_utility_shock,
):
    return jnp.log(consumption) + additive_utility_shock - delta * working


def phelps_deaton_utility(consumption, working, delta):
    return jnp.log(consumption) - delta * working


def intensive_margin_utility(consumption, disutility_of_work):
    return jnp.log(consumption) - disutility_of_work


def disutility_of_work(working_hours, disutility_part_time, disutility_full_time):
    disutility_by_labor_supply = dict(
        zip(
            intensive_margin_working_hours_cats,
            [0, disutility_part_time, disutility_full_time],
            strict=True,
        ),
    )
    return disutility_by_labor_supply[working_hours]


def phelps_deaton_utility_with_filter(
    consumption,
    working,
    delta,
    lagged_retirement,  # noqa: ARG001
):
    return jnp.log(consumption) - delta * working


def working(retirement):
    return 1 - retirement


@lcm.mark.stochastic
def next_wage_category(wage_category, wage_category_transition):
    pass


# Question: common term for _probabilities and _transition?
@lcm.mark.stochastic
def wage_category(wage_category_probabilities):
    pass


def next_human_capital(
    human_capital,
    depreciation,
    working_hours,
    experience_factor_part_time,
):
    """Accumulate human capital (full time equivalent years of experience).

    To discuss:
    - Depreciation needed here? (without it interpretable as years of experience)

    """
    additional_experience = dict(
        zip(
            intensive_margin_working_hours_cats,
            [0, experience_factor_part_time, 1],
            strict=True,
        ),
    )
    return human_capital * depreciation + additional_experience[working_hours]


def intensive_margin_wage(
    human_capital,
    wage_shock_component,
    wage_gamma_0,
    wage_gamma_1,
):
    """Hourly wage rate depending on human capital and a permanent shock component.

    To discuss:
    - Quadratic relationship between human capital and wage rate?
    """
    return np.exp(wage_gamma_0 + wage_gamma_1 * human_capital + wage_shock_component)


@lcm.mark.stochastic(transition="wage_shock_component")
def next_wage_shock_component(wage_shock_component):
    pass


def wage(wage_category, wage_by_category):
    return wage_by_category[wage_category]


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + wage * working


def intensive_margin_next_wealth(
    wealth,
    consumption,
    working_hours,
    wage,
    interest_rate,
):
    return (1 + interest_rate) * (wealth - consumption) + wage * working_hours


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


def age(period):
    return period + 18


def mandatory_retirement_filter(retirement, age):
    return jnp.logical_or(retirement == 1, age < RETIREMENT_AGE)


def absorbing_retirement_filter(retirement, lagged_retirement):
    return jnp.logical_or(retirement == 1, lagged_retirement == 0)


PHELPS_DEATON = {
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
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
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
    },
    "n_periods": 20,
}

PHELPS_DEATON_WITH_PERSISTENT_DISCRETE_TRANSITION_SHOCKS = {
    **PHELPS_DEATON,
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "wage": wage,
        "next_wage_category": next_wage_category,
        "consumption_constraint": consumption_constraint,
        "working": working,
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
        "wage_category": {"options": ["low", "high"]},
        # Question: Possibility to specify number of values in params file?
    },
}

PHELPS_DEATON_WITH_TRANSITORY_DISCRETE_TRANSITION_SHOCKS = {
    **PHELPS_DEATON,
    "functions": {
        "utility": phelps_deaton_utility,
        "next_wealth": next_wealth,
        "wage": wage,
        "wage_category": wage_category,
        "consumption_constraint": consumption_constraint,
        "working": working,
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": N_STATE_GRID_POINTS,
        },
        # Question: Where are options of wage_category specified?
    },
    "shocks": {"wage_category": {"options": ["low", "high"]}},
}

PHELPS_DEATON_WITH_FILTERS = {
    "functions": {
        "utility": phelps_deaton_utility_with_filter,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "next_lagged_retirement": lambda retirement: retirement,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
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
        "lagged_retirement": {"options": [0, 1]},
    },
    "n_periods": 20,
}


# Simple model which incorporates a discrete labor supply choice of three options:
# working full time, working part time, and not working.
INTENSIVE_MARGIN_LABOR_SUPPLY = {
    "functions": {
        "utility": intensive_margin_utility,
        "next_wealth": next_wealth,
        "next_human_capital": next_human_capital,
        "next_wage_shock_component": next_wage_shock_component,
        "consumption_constraint": consumption_constraint,
        "disutility_of_work": disutility_of_work,
        "intensive_margin_wage": intensive_margin_wage,
    },
    "choices": {
        "working_hours": {"options": intensive_margin_working_hours_cats},
        "consumption": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
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
        "human_capital": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 45,
            "n_points": N_STATE_GRID_POINTS,
        },
        "wage_shock_component": {"options": [-1, 0, 1]},
    },
    "n_periods": 20,
}
