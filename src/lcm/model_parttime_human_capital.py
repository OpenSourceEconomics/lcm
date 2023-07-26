"""Labor supply model specification.

Roughly follows Jakobsen, Jorgensen, Low (2022).

Still missing relative to full model:
- no partner (no transition)
- no children (no transition)
- no shocks
- tax and transfer system strongly simplified
- no heterogeneity

Think about Timing

"""
import jax.numpy as jnp

WORKING_HOURS_CATS = [0, 20, 40]

# ======================================================================================
# Utility
# ======================================================================================


def utility(
    crra_value_of_consumption,
    disutility_of_working,
):
    return (
        crra_value_of_consumption
        + disutility_of_working
        # + child_related_disutility_of_working
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
    disutility_of_working_constant,
    disutility_of_working_age,
):
    """Disutility of working.

    The constant depends on working hours.

    The age term increases disutility further (relative to constant and independent of
    working hours).

    """
    return disutility_of_working_constant[working_hours] * (
        1 + disutility_of_working_age * age
    )


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
    additional_experience = dict(
        zip(
            WORKING_HOURS_CATS,
            [0, experience_factor_part_time, 1],
            strict=True,
        ),
    )
    return human_capital * depreciation + additional_experience[working_hours]


# ======================================================================================
# Income and Wealth (Budget Constraint)
# ======================================================================================


def disposable_household_income(labor_income, consumption_floor):
    hh_income = labor_income
    return max(consumption_floor, 0.8 * hh_income)


def next_wealth(
    wealth,
    consumption,
    disposable_household_income,
    interest_rate,
):
    return interest_rate * wealth + disposable_household_income - consumption


def next_wealth_constraint(next_wealth):
    return next_wealth >= 0


# ======================================================================================
# Model Specification
# ======================================================================================


PARTTIME_HUMAN_CAPITAL = {
    "functions": {
        "utility": utility,
        "crra_value_of_consumption": crra_value_of_consumption,
        "working_hours_full_time_equivalent": working_hours_full_time_equivalent,
        "disutility_of_working": disutility_of_working,
        "labor_income": labor_income,
        "wage": wage,
        "next_human_capital": next_human_capital,
        "disposable_household_income": disposable_household_income,
        "next_wealth": next_wealth,
        "next_wealth_constraint": next_wealth_constraint,
    },
    "choices": {
        "working_hours": {"options": WORKING_HOURS_CATS},
        "consumption": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 100,
            "n_points": 11,
        },
    },
    "states": {
        "wealth": {"grid_type": "linspace", "start": 0, "stop": 100, "n_points": 11},
    },
}
