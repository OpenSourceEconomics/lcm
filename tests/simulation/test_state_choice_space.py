import jax.numpy as jnp
from numpy.testing import assert_array_equal

from lcm.input_processing import process_model
from lcm.simulation.state_choice_space import create_state_choice_space
from tests.test_models import get_model_config


def test_create_state_choice_space():
    model_config = get_model_config("iskhakov_et_al_2017", n_periods=3)
    model = process_model(model_config)
    got_space = create_state_choice_space(
        model=model,
        initial_states={
            "wealth": jnp.array([10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 1]),
        },
    )
    assert_array_equal(got_space.discrete_choices["retirement"], jnp.array([0, 1]))
    assert_array_equal(got_space.states["wealth"], jnp.array([10.0, 20.0]))
    assert_array_equal(got_space.states["lagged_retirement"], jnp.array([0, 1]))


def test_create_state_choice_space_replace():
    model_config = get_model_config("iskhakov_et_al_2017", n_periods=3)
    model = process_model(model_config)
    space = create_state_choice_space(
        model=model,
        initial_states={
            "wealth": jnp.array([10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 1]),
        },
    )
    new_space = space.replace(
        states={"wealth": jnp.array([10.0, 30.0])},
    )
    assert_array_equal(new_space.states["wealth"], jnp.array([10.0, 30.0]))
