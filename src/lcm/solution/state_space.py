"""Create a state space for a given model."""

from lcm.interfaces import InternalModel, Space, SpaceInfo


def create_state_choice_space(
    model: InternalModel, *, is_last_period: bool
) -> tuple[Space, SpaceInfo]:
    """Create a state-choice-space for the model.

    A state-choice-space is a compressed representation of all feasible states and the
    feasible discrete choices within that state.

    Args:
        model (Model): A processed model.
        is_last_period (bool): Whether the function is created for the last period.

    Returns:
        Space: Space object containing the sparse and dense variables. This can be used
            to execute a function on an entire space.
        SpaceInfo: A SpaceInfo object that contains all information needed to work with
            the output of a function evaluated on the space.

    """
    # ==================================================================================
    # preparations
    # ==================================================================================
    vi = model.variable_info
    if is_last_period:
        vi = vi.query("~is_auxiliary")

    # ==================================================================================
    # create state choice space
    # ==================================================================================
    _value_grid = _create_value_grid(
        grids=model.grids,
        subset=vi.query("~(is_choice & is_continuous)").index.tolist(),
    )

    state_choice_space = Space(
        sparse_vars={},
        dense_vars=_value_grid,
    )

    # ==================================================================================
    # create state space info
    # ==================================================================================
    # axis_names
    axis_names = vi.query("is_state").index.tolist()

    # lookup_info
    _discrete_states = set(vi.query("is_discrete & is_state").index.tolist())
    lookup_info = {k: v for k, v in model.gridspecs.items() if k in _discrete_states}

    # interpolation info
    _cont_states = set(vi.query("is_continuous & is_state").index.tolist())
    interpolation_info = {k: v for k, v in model.gridspecs.items() if k in _cont_states}

    # indexer infos
    indexer_infos = []  # type: ignore[var-annotated]

    space_info = SpaceInfo(
        axis_names=axis_names,
        lookup_info=lookup_info,  # type: ignore[arg-type]
        interpolation_info=interpolation_info,  # type: ignore[arg-type]
        indexer_infos=indexer_infos,
    )

    return state_choice_space, space_info


def _create_value_grid(grids, subset):
    return {name: grid for name, grid in grids.items() if name in subset}
