from jax import Array

from lcm.interfaces import InternalModel, StateChoiceSpace


def create_state_choice_space(
    model: InternalModel,
    initial_states: dict[str, Array],
) -> StateChoiceSpace:
    """Create the initial state choice space.

    In comparison to the solution, the state choice space in the simulation must be
    created during each iteration, because the states change over time.

    Args:
        model: Model instance.
        initial_states: Dict with initial states.

    Returns:
        State choice space.

    Raises:
        ValueError: If the initial states do not match the state variables in the model.

    """
    vi = model.variable_info
    state_names = set(vi.query("is_state").index)

    if state_names != set(initial_states.keys()):
        missing = state_names - set(initial_states.keys())
        too_many = set(initial_states.keys()) - state_names
        raise ValueError(
            "You need to provide an initial value for each state variable in the model."
            f"\n\nMissing initial states: {missing}\n",
            f"Provided variables that are not states: {too_many}",
        )

    ordered_var_names = tuple(vi.query("is_state | is_discrete").index)
    discrete_choice_names = vi.query("is_choice & is_discrete").index

    discrete_choices = {
        name: grid
        for name, grid in model.grids.items()
        if name in discrete_choice_names
    }

    return StateChoiceSpace(
        states=initial_states,
        choices=discrete_choices,
        ordered_var_names=ordered_var_names,
    )
