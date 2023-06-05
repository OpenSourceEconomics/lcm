import jax.numpy as jnp
import pytest
from jax import random
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON, PHELPS_DEATON_WITH_FILTERS
from lcm.process_model import process_model
from lcm.simulate import (
    _retrieve_non_sparse_choices,
    create_choice_segments,
    create_data_state_choice_space,
    dict_product,
    select_cont_choice_argmax_given_dense_argmax,
)
from numpy.testing import assert_array_equal

# ======================================================================================
# Simulate
# ======================================================================================


@pytest.fixture()
def phelps_deaton_three_period_solution():
    user_model = {**PHELPS_DEATON, "n_periods": 3}

    # solve model
    solve_model, params_template = get_lcm_function(model=user_model)

    # set parameters
    params = params_template.copy()
    params["beta"] = 0.95
    params["utility"]["delta"] = 1.0
    params["next_wealth"]["interest_rate"] = 1 / 0.95 - 1
    params["next_wealth"]["wage"] = 20.0

    vf_arr_list = solve_model(params)
    return vf_arr_list, params


def test_simulate_has_same_value_as_solution(phelps_deaton_three_period_solution):
    vf_arr_list, params = phelps_deaton_three_period_solution

    simulate_model, _ = get_lcm_function(model=PHELPS_DEATON, targets="simulate")

    res = simulate_model(
        params,
        vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([20, 40, 60, 100.0]),
        },
    )
    wealth_grid_index = jnp.array([2, 4, 6, 10])
    assert jnp.all(vf_arr_list[0][wealth_grid_index] == res[0]["value"])


def test_simulate_correct_choices(phelps_deaton_three_period_solution):
    vf_arr_list, params = phelps_deaton_three_period_solution

    simulate_model, _ = get_lcm_function(model=PHELPS_DEATON, targets="simulate")

    res = simulate_model(
        params,
        vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([20, 40, 60, 100.0]),
        },
    )

    # assert that value is increasing in initial wealth
    for period in range(3):
        assert jnp.all(jnp.diff(res[period]["value"]) >= 0)

    # assert that no one works in the last period
    assert jnp.all(res[2]["choices"]["retirement"] == 1)


# ======================================================================================
# Helper functions
# ======================================================================================


def test_retrieve_non_sparse_choices():
    got = _retrieve_non_sparse_choices(
        index=jnp.array([0, 3, 7]),
        grids={"a": jnp.linspace(0, 1, 5), "b": jnp.linspace(10, 20, 6)},
        grid_shape=(5, 6),
    )
    assert_array_equal(got["a"], jnp.array([0, 0, 0.25]))
    assert_array_equal(got["b"], jnp.array([10, 16, 12]))


def test_select_cont_choice_argmax_given_dense_argmax():
    ccc_argmax = jnp.array(
        [
            [0, 1],
            [1, 0],
        ],
    )
    dense_argmax = jnp.array([0, 1])
    dense_vars_grid_shape = (2,)
    got = select_cont_choice_argmax_given_dense_argmax(
        conditional_cont_choice_argmax=ccc_argmax,
        dense_argmax=dense_argmax,
        dense_vars_grid_shape=dense_vars_grid_shape,
    )
    assert jnp.all(got == jnp.array([0, 0]))


def test_create_data_state_choice_space():
    model = process_model(PHELPS_DEATON_WITH_FILTERS)
    got_space, got_segment_info = create_data_state_choice_space(
        states={
            "wealth": jnp.array([10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 1]),
        },
        model=model,
    )
    assert got_space.dense_vars == {}
    assert_array_equal(got_space.sparse_vars["wealth"], jnp.array([10.0, 10.0, 20.0]))
    assert_array_equal(got_space.sparse_vars["lagged_retirement"], jnp.array([0, 0, 1]))
    assert_array_equal(got_space.sparse_vars["retirement"], jnp.array([0, 1, 1]))
    assert_array_equal(got_segment_info["segment_ids"], jnp.array([0, 0, 1]))
    assert got_segment_info["num_segments"] == 2


def test_choice_segments():
    got = create_choice_segments(
        mask=jnp.array([True, False, True, False, True, False]),
        n_sparse_states=2,
    )
    assert_array_equal(jnp.array([0, 0, 1]), got["segment_ids"])
    assert got["num_segments"] == 2


def test_choice_segments_weakly_increasing():
    key = random.PRNGKey(12345)
    n_states, n_choices = random.randint(key, shape=(2,), minval=1, maxval=100)
    mask_len = n_states * n_choices
    mask = random.choice(key, a=2, shape=(mask_len,), p=jnp.array([0.5, 0.5]))
    got = create_choice_segments(mask, n_sparse_states=n_states)["segment_ids"]
    assert jnp.all(got[1:] - got[:-1] >= 0)


def test_dict_product():
    d = {"a": jnp.array([0, 1]), "b": jnp.array([2, 3])}
    got_dict, got_length = dict_product(d)
    exp = {"a": jnp.array([0, 0, 1, 1]), "b": jnp.array([2, 3, 2, 3])}
    assert got_length == 4
    for key, val in exp.items():
        assert_array_equal(got_dict[key], val)
