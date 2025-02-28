"""Create a state space for a given model."""

import pandas as pd
from jax import Array

from lcm.grids import ContinuousGrid, DiscreteGrid
from lcm.interfaces import InternalModel, StateChoiceSpace, StateSpaceInfo


def create_state_choice_space(
    model: InternalModel,
    *,
    initial_states: dict[str, Array] | None = None,
    is_last_period: bool = False,
) -> StateChoiceSpace:
    """Create a state-choice-space.

    Creates the state-choice-space for the solution and simulation of a model. In the
    simulation, initial states must be provided.

    Args:
        model: A processed model.
        initial_states: A dictionary with the initial values of the state variables.
            If None, the initial values are the minimum values of the state variables.
        is_last_period: Whether the state-choice-space is created for the last period,
            in which case auxiliary variables are not included.

    Returns:
        A state-choice-space. Contains the grids of the discrete and continuous choices,
        the grids of the state variables, or the initial values of the state variables,
        and the names of the state and choice variables in the order they appear in the
        variable info table.

    """
    vi = model.variable_info
    if is_last_period:
        vi = vi.query("~is_auxiliary")

    if initial_states is None:
        states = {sn: model.grids[sn] for sn in vi.query("is_state").index}
    else:
        _validate_initial_states(initial_states, variable_info=vi)
        states = initial_states

    discrete_choices = {
        name: model.grids[name] for name in vi.query("is_choice & is_discrete").index
    }
    continuous_choices = {
        name: model.grids[name] for name in vi.query("is_choice & is_continuous").index
    }
    ordered_var_names = tuple(vi.query("is_state | is_discrete").index)

    return StateChoiceSpace(
        states=states,
        discrete_choices=discrete_choices,
        continuous_choices=continuous_choices,
        ordered_var_names=ordered_var_names,
    )


def create_state_space_info(
    model: InternalModel,
    *,
    is_last_period: bool,
) -> StateSpaceInfo:
    """Create a state-space information for the model solution.

    A state-space information is a compressed representation of all feasible states.

    Args:
        model: A processed model.
        is_last_period: Whether the function is created for the last period.

    Returns:
        The state-space information.

    """
    vi = model.variable_info
    if is_last_period:
        vi = vi.query("~is_auxiliary")

    state_names = vi.query("is_state").index.tolist()

    discrete_states = {
        name: grid_spec
        for name, grid_spec in model.gridspecs.items()
        if name in state_names and isinstance(grid_spec, DiscreteGrid)
    }

    continuous_states = {
        name: grid_spec
        for name, grid_spec in model.gridspecs.items()
        if name in state_names and isinstance(grid_spec, ContinuousGrid)
    }

    return StateSpaceInfo(
        states_names=tuple(state_names),
        discrete_states=discrete_states,
        continuous_states=continuous_states,
    )


def _validate_initial_states(
    initial_states: dict[str, Array], variable_info: pd.DataFrame
) -> None:
    """Checks if each model-state has an initial value."""
    states_names_in_model = set(variable_info.query("is_state").index)
    provided_states_names = set(initial_states)

    if states_names_in_model != provided_states_names:
        missing = states_names_in_model - provided_states_names
        too_many = provided_states_names - states_names_in_model
        raise ValueError(
            "You need to provide an initial array for each state variable in the model."
            f"\n\nMissing initial states: {missing}\n",
            f"Provided variables that are not states: {too_many}",
        )
