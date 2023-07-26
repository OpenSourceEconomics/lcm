"""Labor supply model specification.

Roughly follows Jakobsen, Jorgensen, Low (2022).

"""
import jax.numpy as jnp

WORKING_HOURS_CATS = [0, 20, 40]

# ======================================================================================
# Utility
# ======================================================================================


def utility(
    crra_value_of_consumption,
    disutility_of_working,
    child_related_disutility_of_working,
):
    return (
        crra_value_of_consumption
        + disutility_of_working
        + child_related_disutility_of_working
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


def child_related_disutility_of_working(
    working_hours,
    n_children,
    age_youngest_child,
    disutility_of_working_constant,
    child_related_disutility_of_working_child,
    child_related_disutility_of_working_n_children,
    child_related_disutility_of_working_age_0,
    child_related_disutility_of_working_age_1,
    child_related_disutility_of_working_age_2,
    child_related_disutility_of_working_type_factor,
    gender_attitude_type,
):
    relative_increase_disutil = (
        # At least one child
        (n_children > 0) * child_related_disutility_of_working_child
        # Number of children
        + (n_children - 1) * child_related_disutility_of_working_n_children
        # Additional term for very young children
        + (age_youngest_child == 0) * child_related_disutility_of_working_age_0
        + (age_youngest_child == 1) * child_related_disutility_of_working_age_1
        + (age_youngest_child == 2)  # noqa: PLR2004
        * child_related_disutility_of_working_age_2
    )

    # Assumption: Type has same effect on all components of child-related disutility
    # Assumption: No difference between full-time and part-time work with respect to
    # child-related disutility other than through constant
    return (
        disutility_of_working_constant[working_hours]
        * relative_increase_disutil
        * child_related_disutility_of_working_type_factor[gender_attitude_type]
    )


# ======================================================================================
# Labor Income
# ======================================================================================


def labor_income(
    working_hours,
    wage,
):
    return working_hours * wage


def labor_income_partner(wage_partner):
    """Labor income of the partner.

    Assumption: The partner always works full-time.
    """
    return WORKING_HOURS_CATS[-1] * wage_partner


def wage_partner(
    has_partner,
    age,
    wage_partner_constant,
    wage_partner_age,
    wage_partner_age_squared,
    gender_attitude_type,
):
    """Wage of the partner. It depends on the female's state variables.

    We should add some kind of shock, at some point (either employment shock or wage
    shock or both).

    """
    if has_partner:
        out = (
            wage_partner_constant[gender_attitude_type]
            + age * wage_partner_age[gender_attitude_type]
            + age**2 * wage_partner_age_squared[gender_attitude_type]
        )
    else:
        out = 0
    return out


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
        "utility": utility,
        "next_wealth": next_wealth,
        "next_wealth_constraint": next_wealth_constraint,
    },
    "choices": {
        "working_hours": {"options": WORKING_HOURS_CATS},
    },
    "states": {
        "job_offer": {"options": [0, 1]},
        "wealth": {"grid_type": "linspace", "start": 0, "stop": 100, "n_points": 11},
    },
    "shocks": {
        "human_capital_shock": "lognormal",
    },
}
