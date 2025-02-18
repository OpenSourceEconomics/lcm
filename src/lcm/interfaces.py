from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from jax import Array

from lcm.grids import ContinuousGrid, DiscreteGrid, Grid
from lcm.typing import InternalUserFunction, ParamsDict, ShockType


@dataclass(frozen=True)
class StateChoiceSpace:
    """The state-choice space.

    When used for the model solution:
    ---------------------------------

    The state-choice space becomes the full Cartesian product of the state variables and
    the choice variables.

    When used for the simulation:
    ----------------------------

    The state-choice space becomes the product of state-combinations with the full
    Cartesian product of the choice variables.

    Attributes:
        states: Dictionary containing the values of the state variables.
        choices: Dictionary containing the values of the choice variables.
        ordered_var_names: Tuple with names of state and choice variables in the order
            they appear in the variable info table.

    """

    states: dict[str, Array]
    choices: dict[str, Array]
    ordered_var_names: tuple[str, ...]


@dataclass(frozen=True)
class StateSpaceInfo:
    """Information to work with the output of a function evaluated on a state space.

    An example is the value function array, which is the output of the value function
    evaluated on the state space.

    Attributes:
        var_names: Tuple with names of state variables.
        discrete_vars: Dictionary with grids of discrete state variables.
        continuous_vars: Dictionary with grids of continuous state variables.

    """

    states_names: tuple[str, ...]
    discrete_states: Mapping[str, DiscreteGrid]
    continuous_states: Mapping[str, ContinuousGrid]


@dataclass(frozen=True)
class InternalModel:
    """Internal representation of a user model.

    Attributes:
        grids: Dictionary that maps names of model variables to grids of feasible values
            for that variable.
        gridspecs: Dictionary that maps names of model variables to specifications from
            which grids of feasible values can be built.
        variable_info: A table with information about all variables in the model. The
            index contains the name of a model variable. The columns are booleans that
            are True if the variable has the corresponding property. The columns are:
            is_state, is_choice, is_continuous, is_discrete.
        functions: Dictionary that maps names of functions to functions. The functions
            differ from the user functions in that they take `params` as a keyword
            argument. Two cases:
            - If the original function depended on model parameters, those are
              automatically extracted from `params` and passed to the original
              function.
            - Otherwise, the `params` argument is simply ignored.
        function_info: A table with information about all functions in the model. The
            index contains the name of a function. The columns are booleans that are
            True if the function has the corresponding property. The columns are:
            is_constraint, is_next.
        params: Dict of model parameters.
        n_periods: Number of periods.
        random_utility_shocks: Type of random utility shocks.

    """

    grids: dict[str, Array]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    functions: dict[str, InternalUserFunction]
    function_info: pd.DataFrame
    params: ParamsDict
    n_periods: int
    # Not properly processed yet
    random_utility_shocks: ShockType
