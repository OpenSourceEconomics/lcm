"""Example specifications of fully discrete deterministic consumption-saving model.

The specification builds on the example model presented in the paper: "The endogenous
grid method for discrete-continuous dynamic choice models with (or without) taste
shocks" by Fedor Iskhakov, Thomas H. Jørgensen, John Rust and Bertel Schjerning (2017,
https://doi.org/10.3982/QE643). See module `tests.test_models.deterministic` for the
continuous version.

"""

from dataclasses import dataclass

import jax.numpy as jnp

from lcm import DiscreteGrid, Model
from tests.test_models.deterministic import (
    RetirementStatus,
    labor_income,
    next_wealth,
    utility,
    working,
)

# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class ConsumptionChoice:
    low: int = 0
    high: int = 1


@dataclass
class WealthStatus:
    low: int = 0
    medium: int = 1
    high: int = 2


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def utility_discrete(consumption, working, disutility_of_work):
    # In the discrete model, consumption is defined as "low" or "high". This can be
    # translated to the levels 1 and 2.
    consumption_level = 1 + (consumption == ConsumptionChoice.high)
    return utility(consumption_level, working, disutility_of_work)


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth_discrete(wealth, consumption, labor_income, interest_rate):
    # For discrete state variables, we need to assure that the next state is also a
    # valid state, i.e., it is a member of the discrete grid.
    continuous = next_wealth(wealth, consumption, labor_income, interest_rate)
    return jnp.clip(jnp.rint(continuous), WealthStatus.low, WealthStatus.high).astype(
        jnp.int32
    )


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def consumption_constraint(consumption, wealth):
    return consumption <= wealth


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
def absorbing_retirement_filter(retirement, lagged_retirement):
    return jnp.logical_or(
        retirement == RetirementStatus.retired,
        lagged_retirement == RetirementStatus.working,
    )


# ======================================================================================
# Model specifications
# ======================================================================================
ISKHAKOV_ET_AL_2017_DISCRETE = Model(
    description=(
        "Starts from Iskhakov et al. (2017), removes filters and the lagged_retirement "
        "state, and makes the consumption decision and the wealth state discrete."
    ),
    n_periods=3,
    functions={
        "utility": utility_discrete,
        "next_wealth": next_wealth_discrete,
        "consumption_constraint": consumption_constraint,
        "labor_income": labor_income,
        "working": working,
    },
    choices={
        "retirement": DiscreteGrid(RetirementStatus),
        "consumption": DiscreteGrid(ConsumptionChoice),
    },
    states={
        "wealth": DiscreteGrid(WealthStatus),
    },
)
