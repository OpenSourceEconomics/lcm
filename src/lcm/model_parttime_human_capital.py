"""Labor supply model specification.

Roughly follows Jakobsen, Jorgensen, Low (2022).

Still missing relative to full model:
- no partner (no transition)
- no children (no transition)
- no shocks
- tax and transfer system strongly simplified
- no heterogeneity
- retirement periods (age >= 60 -> retired | fix continuation value at 60)

Think about Timing

"""
import jax.numpy as jnp

PART_TIME_WORKING_HOURS = 20
FULL_TIME_WORKING_HOURS = 40

WORKING_HOURS_CATS = [0, PART_TIME_WORKING_HOURS, FULL_TIME_WORKING_HOURS]

# ======================================================================================
# Utility
# ======================================================================================


def utility(
    crra_value_of_consumption,
    disutility_of_working,
):
    """Utility.

    todo:
    - add child_related_disutility_of_working
    
    """
    return (
        crra_value_of_consumption
        + disutility_of_working
    )


def crra_value_of_consumption(
    consumption,
    has_partner,
    n_children,
    crra_coefficient,
):
    return (
        (consumption / (1 + 0.5 * has_partner + 0.3 * n_children))
        ** (1 - crra_coefficient)
    ) / (1 - crra_coefficient)


def working_hours_full_time_equivalent(working_hours):
    return working_hours / WORKING_HOURS_CATS[-1]


def disutility_of_working(
    working_hours,
    age,
    disutility_of_working_fulltime,
    disutility_of_working_parttime,
    disutility_of_working_age,
):
    """Disutility of working.

    The constant depends on working hours.

    The age term increases disutility further (relative to constant and independent of
    working hours).

    """
    disutility_age_factor = 1 + disutility_of_working_age * age

    disutility_of_working = jnp.where(
        working_hours == FULL_TIME_WORKING_HOURS,
        disutility_of_working_fulltime,
        disutility_of_working_parttime,
    )

    disutility_of_working = jnp.where(
        working_hours == 0,
        0,
        disutility_of_working,
    )

    return disutility_of_working * disutility_age_factor


def age(period):
    return period + 18


# ======================================================================================
# Labor Income
# ======================================================================================


def labor_income(
    working_hours,
    wage,
):
    return working_hours * wage


def wage(
    human_capital,
    wage_coefficient_constant,
    wage_coefficient_per_human_capital,
):
    log_wage = (
        wage_coefficient_constant + wage_coefficient_per_human_capital * human_capital
    )
    return jnp.exp(log_wage)


def wage_constraint(wage, human_capital):
    return jnp.logical_and(wage > 0, human_capital > 0)


# Human capital state transition
# ======================================================================================


def next_human_capital(
    human_capital,
    depreciation,
    working_hours,
    experience_factor_part_time,
):
    """Accumulate human capital (full time equivalent years of experience).

    To discuss:
    - Depreciation needed here? (without it interpretable as years of experience)

    To add:  
    - shock

    """
    additional_experience = jnp.where(
        working_hours == FULL_TIME_WORKING_HOURS,
        1,
        experience_factor_part_time,
    )

    additional_experience = jnp.where(
        working_hours == 0,
        0,
        additional_experience,
    )

    return human_capital * (1 - depreciation) + additional_experience


# ======================================================================================
# Income and Wealth (Budget Constraint)
# ======================================================================================


def disposable_household_income(labor_income, income_floor):
    hh_income = labor_income
    return jnp.maximum(income_floor, 0.8 * hh_income)


def next_wealth(
    wealth,
    consumption,
    disposable_household_income,
    interest_rate,
):
    return interest_rate * wealth + disposable_household_income - consumption


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


# ======================================================================================
# Model Specification
# ======================================================================================


PARTTIME_HUMAN_CAPITAL = {
    "n_periods": 3,
    "functions": {
        "utility": utility,
        "crra_value_of_consumption": crra_value_of_consumption,
        "working_hours_full_time_equivalent": working_hours_full_time_equivalent,
        "disutility_of_working": disutility_of_working,
        "labor_income": labor_income,
        "wage": wage,
        "disposable_household_income": disposable_household_income,
        "next_human_capital": next_human_capital,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "age": age,
        "wage_constraint": wage_constraint,
    },
    "choices": {
        "working_hours": {"options": WORKING_HOURS_CATS},
        "consumption": {
            "grid_type": "linspace",
            "start": 100,
            "stop": 10_000,
            "n_points": 200,
        },
    },
    "states": {
        "human_capital": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 3,
            "n_points": 10,
        },
        "wealth": {
            "grid_type": "linspace",
            "start": 100,
            "stop": 10_000,
            "n_points": 200,
        },
    },
}

PARTTIME_HUMAN_CAPITAL_PARAMS = {
    "beta": 0.98,
    "utility": {},
    "age": {"period": 0},
    "crra_value_of_consumption": {
        "crra_coefficient": 1.5,  # jacobsen et al, attanasio & weber
        "has_partner": 0.0,
        "n_children": 0.0,
    },
    "working_hours_full_time_equivalent": {},
    "disutility_of_working": {  # jacobsen et al (motivated)
        "disutility_of_working_age": -0.02,
        "disutility_of_working_fulltime": -0.5,
        "disutility_of_working_parttime": -0.3,
    },
    "wage": {  # jacobsen et al
        "wage_coefficient_constant": 0.8,
        "wage_coefficient_per_human_capital": 0.09,
    },
    "disposable_household_income": {"income_floor": 1000.0},
    "next_wealth": {"interest_rate": 1.03},
    "consumption_constraint": {},
    "next_human_capital": {
        "depreciation": 0.9,
        "experience_factor_part_time": 0.5,
    },
    "wage_constraint": {},
}
