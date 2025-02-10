"""Create a state space for a given model."""

from lcm.interfaces import InternalModel, SolutionSpace, SpaceInfo


def create_state_choice_space(
    model: InternalModel,
    *,
    is_last_period: bool,
) -> SolutionSpace:
    """Create a state-choice-space for the model solution.

    A state-choice-space is a compressed representation of all feasible states and the
    feasible discrete choices within that state.

    Args:
        model: A processed model.
        is_last_period: Whether the function is created for the last period.

    Returns:
        SolutionSpace: An object containing the variable values of all variables in the
            state-choice-space, the grid specifications for the state variables, and the
            names of the state variables. Continuous choice variables are not included.

    """
    vi = model.variable_info
    if is_last_period:
        vi = vi.query("~is_auxiliary")

    discrete_states_names = vi.query("is_discrete & is_state").index.tolist()
    continuous_states_names = vi.query("is_continuous & is_state").index.tolist()

    discrete_states = {sn: model.gridspecs[sn] for sn in discrete_states_names}
    continuous_states = {sn: model.gridspecs[sn] for sn in continuous_states_names}

    # Create a dictionary with all state and choice variables and their feasible values,
    # except for continuous choice variables, since they are treated differently.
    space_grids = {
        sn: model.grids[sn] for sn in vi.query("is_state | is_discrete").index.tolist()
    }

    state_space_info = SpaceInfo(
        axis_names=discrete_states_names + continuous_states_names,
        lookup_info=discrete_states,  # type: ignore[arg-type]
        interpolation_info=continuous_states,  # type: ignore[arg-type]
    )

    return SolutionSpace(
        vars=space_grids,
        state_space_info=state_space_info,
    )
