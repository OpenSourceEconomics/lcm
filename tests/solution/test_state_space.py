import jax.numpy as jnp

from lcm.input_processing import process_model
from lcm.interfaces import StateChoiceSpace, StateSpaceInfo
from lcm.solution.state_choice_space import (
    create_state_choice_space,
    create_state_space_info,
)
from tests.test_models import get_model_config


def test_create_state_choice_space():
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    internal_model = process_model(model)

    state_choice_space = create_state_choice_space(
        model=internal_model,
        is_last_period=False,
    )

    assert isinstance(state_choice_space, StateChoiceSpace)
    assert jnp.array_equal(
        state_choice_space.discrete_choices["retirement"],
        model.choices["retirement"].to_jax(),
    )
    assert jnp.array_equal(
        state_choice_space.states["wealth"], model.states["wealth"].to_jax()
    )


def test_create_state_space_info():
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    internal_model = process_model(model)

    state_space_info = create_state_space_info(
        model=internal_model,
        is_last_period=False,
    )

    assert isinstance(state_space_info, StateSpaceInfo)
    assert state_space_info.states_names == ("wealth",)
    assert state_space_info.discrete_states == {}
    assert state_space_info.continuous_states == model.states
