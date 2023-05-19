import jax.numpy as jnp
from lcm.example_models import PHELPS_DEATON_WITH_FILTERS
from lcm.process_model import process_model
from lcm.simulate import create_data_state_choice_space, dict_product
from numpy.testing import assert_array_equal


def test_create_sata_state_choice_space():
    model = process_model(PHELPS_DEATON_WITH_FILTERS)
    create_data_state_choice_space(
        initial_states={
            "wealth": jnp.array([0.0, 10.0, 10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 0, 1, 1]),
        },
        model=model,
    )


def test_dict_product():
    d = {"a": jnp.array([0, 1]), "b": jnp.array([2, 3])}
    got_dict, got_length = dict_product(d)
    exp = {"a": jnp.array([0, 0, 1, 1]), "b": jnp.array([2, 3, 2, 3])}
    assert got_length == 4
    for key, val in exp.items():
        assert_array_equal(got_dict[key], val)
