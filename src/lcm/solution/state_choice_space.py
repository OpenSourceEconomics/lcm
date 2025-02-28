"""Create a state space for a given model."""

from lcm.interfaces import InternalModel, StateChoiceSpace, StateSpaceInfo


def create_state_choice_space(
    model: InternalModel,
    *,
    is_last_period: bool,
) -> StateChoiceSpace:
    """Create a state-choice-space for the model solution.

    A state-choice-space is a compressed representation of all feasible states and the
    feasible discrete choices within that state.

    Args:
        model: A processed model.
        is_last_period: Whether the function is created for the last period.

    Returns:
        - An object containing the variable values of all variables in the
          state-choice-space, the grid specifications for the state variables, and the
          names of the state variables. Continuous choice variables are not included.
        - The state-space information.

    """
    vi = model.variable_info
    if is_last_period:
        vi = vi.query("~is_auxiliary")

    state_grids = {sn: model.grids[sn] for sn in vi.query("is_state").index}
    discrete_choices = {
        sn: model.grids[sn] for sn in vi.query("is_choice & is_discrete").index
    }
    continuous_choices = {
        sn: model.grids[sn] for sn in vi.query("is_choice & is_continuous").index
    }
    ordered_var_names = tuple(vi.query("is_state | is_discrete").index)

    return StateChoiceSpace(
        states=state_grids,
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

    discrete_states_names = vi.query("is_discrete & is_state").index.tolist()
    continuous_states_names = vi.query("is_continuous & is_state").index.tolist()

    discrete_states = {sn: model.gridspecs[sn] for sn in discrete_states_names}
    continuous_states = {sn: model.gridspecs[sn] for sn in continuous_states_names}

    return StateSpaceInfo(
        states_names=tuple(discrete_states_names + continuous_states_names),
        discrete_states=discrete_states,  # type: ignore[arg-type]
        continuous_states=continuous_states,  # type: ignore[arg-type]
    )
