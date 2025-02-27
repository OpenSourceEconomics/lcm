import dataclasses as dc
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd
from jax import Array

from lcm.grids import ContinuousGrid, DiscreteGrid, Grid
from lcm.typing import InternalUserFunction, ParamsDict, ShockType
from lcm.utils import first_non_none


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

    Note:
    -----
    We store discrete and continuous choices separately since these are handled during
    different stages of the solution and simulation processes.

    Attributes:
        states: Dictionary containing the values of the state variables.
        discrete_choices: Dictionary containing the values of the discrete choice
            variables.
        continuous_choices: Dictionary containing the values of the continuous choice
            variables.
        ordered_var_names: Tuple with names of state and choice variables in the order
            they appear in the variable info table.

    """

    states: dict[str, Array]
    discrete_choices: dict[str, Array]
    continuous_choices: dict[str, Array]
    ordered_var_names: tuple[str, ...]

    def replace(
        self,
        states: dict[str, Array] | None = None,
        discrete_choices: dict[str, Array] | None = None,
        continuous_choices: dict[str, Array] | None = None,
    ) -> "StateChoiceSpace":
        """Replace the states or choices in the state-choice space.

        Args:
            states: Dictionary with new states. If None, the existing states are used.
            discrete_choices: Dictionary with new discrete choices. If None, the
                existing discrete choices are used.
            continuous_choices: Dictionary with new continuous choices. If None, the
                existing continuous choices are used.

        Returns:
            New state-choice space with the replaced states or choices.

        """
        states = first_non_none(states, self.states)
        discrete_choices = first_non_none(discrete_choices, self.discrete_choices)
        continuous_choices = first_non_none(continuous_choices, self.continuous_choices)
        return dc.replace(
            self,
            states=states,
            discrete_choices=discrete_choices,
            continuous_choices=continuous_choices,
        )


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


@dataclass(frozen=True)
class InternalSimulationPeriodResults:
    """The results of a simulation for one period."""

    value: Array
    choices: dict[str, Array]
    states: dict[str, Array]
