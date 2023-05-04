"""Labor supply model specification.

Roughly follows Jakobsen, Jorgensen, Low (2022).

"""
import jax.numpy as jnp

MIN_AGE_YOUNG_CHILD = 3

# ======================================================================================
# Utility
# ======================================================================================


def utility_household(
    utility_male,
    utility_female,
    female_bargaining_weight,
):
    return (
        female_bargaining_weight * utility_female
        + (1 - female_bargaining_weight) * utility_male
    )


def utility_male(
    crra_value_of_consumption,
    disutility_of_working,
    child_related_disutility_of_working,
):
    return (
        crra_value_of_consumption
        + disutility_of_working
        + child_related_disutility_of_working
    )


def child_related_disutility_of_working(
    working_hours,
    n_children,
    age_youngest_child,
    disutility_of_working_constant,
    child_related_disutility_of_working_child,
    child_related_disutility_of_working_more,
    child_related_disutility_of_working_young,
    gender_attitude_type,
):
    young_child_present = age_youngest_child <= MIN_AGE_YOUNG_CHILD

    child_related_disutility_per_working_hour = disutility_of_working_constant * (
        child_related_disutility_of_working_child[gender_attitude_type]
        + child_related_disutility_of_working_more[gender_attitude_type]
        * (n_children - 1)
        + child_related_disutility_of_working_young[gender_attitude_type]
        * young_child_present
    )

    return working_hours * child_related_disutility_per_working_hour


def disutility_of_working(
    working_hours,
    age,
    disutility_of_working_constant,
    disutility_of_working_age,
    disutility_of_working_age_squared,
):
    return (
        working_hours
        * disutility_of_working_constant
        * (
            1
            + disutility_of_working_age * age
            + disutility_of_working_age_squared * age**2
        )
    )


def crra_value_of_consumption(
    consumption,
    n_children,
    crra_coefficient,
):
    return ((consumption / (1.5 + 0.3 * n_children)) ** (1 - crra_coefficient)) / (
        1 - crra_coefficient
    )


# ======================================================================================
# Labor Income
# ======================================================================================


def labor_income(
    working_hours,
    wage,
):
    return working_hours * wage


def labor_income_female(
    working_hours_female,
    wage_female,
):
    return labor_income(working_hours=working_hours_female, wage=wage_female)


def labor_income_male(
    working_hours_male,
    wage_male,
):
    return labor_income(working_hours=working_hours_male, wage=wage_male)


def wage(
    human_capital,
    wage_coefficient_constant,
    wage_coefficient_per_human_capital,
):
    log_wage = (
        wage_coefficient_constant + wage_coefficient_per_human_capital * human_capital
    )
    return jnp.exp(log_wage)


def wage_female(
    human_capital_female,  # noqa: ARG001
    wage_coefficient_constant_female,  # noqa: ARG001
    wage_coefficient_per_human_capital_female,  # noqa: ARG001
):
    pass


def wage_male(
    human_capital_male,  # noqa: ARG001
    wage_coefficient_constant_male,  # noqa: ARG001
    wage_coefficient_per_human_capital_male,  # noqa: ARG001
):
    pass


# Human capital state transition
# ======================================================================================


def next_human_capital_male(
    human_capital_male,
    working_hours_male,
    human_capital_depreciation,
    human_capital_shock_male,
):
    return next_human_capital(
        human_capital=human_capital_male,
        working_hours=working_hours_male,
        human_capital_depreciation=human_capital_depreciation,
        human_capital_shock=human_capital_shock_male,
    )


def next_human_capital_female(
    human_capital_female,
    working_hours_female,
    human_capital_depreciation,
    human_capital_shock_female,
):
    return next_human_capital(
        human_capital=human_capital_female,
        working_hours=working_hours_female,
        human_capital_depreciation=human_capital_depreciation,
        human_capital_shock=human_capital_shock_female,
    )


def next_human_capital(
    human_capital,
    working_hours,
    human_capital_depreciation,
    human_capital_shock,
):
    return human_capital_shock * (
        (1 - human_capital_depreciation) * human_capital + working_hours
    )


# ======================================================================================
# Child Care Costs
# ======================================================================================


def child_care_costs(
    working_hours_female,
    working_hours_male,
    n_children,
    age_youngest_child,
    labor_income_female,
    labor_income_male,
):
    if n_children == 0 or working_hours_female == 0 or working_hours_male == 0:
        costs = 0

    else:
        costs = potential_child_care_costs(
            n_children=n_children,
            age_youngest_child=age_youngest_child,
            labor_income_female=labor_income_female,
            labor_income_male=labor_income_male,
        )

    return costs


def potential_child_care_costs(
    n_children,  # noqa: ARG001
    age_youngest_child,  # noqa: ARG001
    labor_income_female,  # noqa: ARG001
    labor_income_male,  # noqa: ARG001
):
    pass


# ======================================================================================
# Income and Wealth (Budget Constraint)
# ======================================================================================


def gettsim(*args, **kwargs):  # noqa: ARG001
    pass


def disposable_household_income(labor_income_female, labor_income_male, **kwargs):
    return gettsim(labor_income_female, labor_income_male, **kwargs)


def next_wealth(
    wealth,
    consumption,
    disposable_household_income,
    child_care_costs,
    interest_rate,
):
    return (
        interest_rate * wealth
        + disposable_household_income
        - child_care_costs
        - consumption
    )


def next_wealth_constraint(next_wealth):
    return next_wealth >= 0


# ======================================================================================
# Model Specification
# ======================================================================================


LABOR_SUPPLY = {
    "functions": {
        "utility_household": utility_household,
        "next_wealth": next_wealth,
        "next_wealth_constraint": next_wealth_constraint,
    },
    "choices": {
        "working_hours": {"options": [0, 10, 20, 30, 40]},
    },
    "states": {
        "job_offer": {"options": [0, 1]},
        "wealth": {"grid_type": "linspace", "start": 0, "stop": 100, "n_points": 11},
    },
    "shocks": {
        "human_capital_shock": "lognormal",
    },
}
