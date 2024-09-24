"""Example specification of a stochastic consumption-saving model.

This specification is motivated by the example model presented in the paper: "The
endogenous grid method for discrete-continuous dynamic choice models with (or without)
taste shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning
(2017, https://doi.org/10.3982/QE643).

See also the specifications in tests/test_models/deterministic.py.

"""

from dataclasses import dataclass

import jax.numpy as jnp

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Model

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class HealthStatus:
    bad: int = 0
    good: int = 1


@dataclass
class PartnerStatus:
    single: int = 0
    partnered: int = 1


@dataclass
class WorkingStatus:
    retired: int = 0
    working: int = 1


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    consumption,
    working,
    health,
    # Temporary workaround for bug described in issue #30, which requires us to pass
    # all state variables to the utility function.
    # TODO(@timmens): Remove function arguments once #30 is fixed.
    # https://github.com/OpenSourceEconomics/lcm/issues/30
    partner,  # noqa: ARG001
    disutility_of_work,
):
    return jnp.log(consumption) - (1 - health / 2) * disutility_of_work * working


# --------------------------------------------------------------------------------------
# Auxiliary variables
# --------------------------------------------------------------------------------------
def labor_income(working, wage):
    return working * wage


# --------------------------------------------------------------------------------------
# Deterministic state transitions
# --------------------------------------------------------------------------------------
def next_wealth(wealth, consumption, labor_income, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + labor_income


# --------------------------------------------------------------------------------------
# Stochastic state transitions
# --------------------------------------------------------------------------------------
@lcm.mark.stochastic
def next_health(health, partner):
    pass


@lcm.mark.stochastic
def next_partner(_period, working, partner):
    pass


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth):
    return consumption <= wealth


# ======================================================================================
# Model specification
# ======================================================================================

ISKHAKOV_ET_AL_2017_STOCHASTIC = Model(
    description=(
        "Starts from Iskhakov et al. (2017), removes filters and the lagged_retirement "
        "state, and adds discrete stochastic state variables health and partner."
    ),
    n_periods=3,
    functions={
        "utility": utility,
        "next_wealth": next_wealth,
        "next_health": next_health,
        "next_partner": next_partner,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
    },
    choices={
        "working": DiscreteGrid(WorkingStatus),
        "consumption": LinspaceGrid(
            start=1,
            stop=100,
            n_points=200,
        ),
    },
    states={
        "health": DiscreteGrid(HealthStatus),
        "partner": DiscreteGrid(PartnerStatus),
        "wealth": LinspaceGrid(
            start=1,
            stop=100,
            n_points=100,
        ),
    },
)
