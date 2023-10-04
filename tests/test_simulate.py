import jax.numpy as jnp
import pandas as pd
import pytest
from jax import random
from lcm.entry_point import (
    create_compute_conditional_continuation_policy,
    get_lcm_function,
    get_next_state_function,
)
from lcm.example_models import (
    N_CHOICE_GRID_POINTS,
    PHELPS_DEATON,
    PHELPS_DEATON_WITH_FILTERS,
)
from lcm.model_functions import get_utility_and_feasibility_function
from lcm.process_model import process_model
from lcm.simulate import (
    _retrieve_non_sparse_choices,
    create_choice_segments,
    create_data_scs,
    determine_discrete_dense_choice_axes,
    dict_product,
    filter_ccv_policy,
    simulate,
)
from lcm.state_space import create_state_choice_space
from numpy.testing import assert_array_almost_equal, assert_array_equal

# ======================================================================================
# Test simulate using raw inputs
# ======================================================================================


@pytest.fixture()
def simulate_inputs():
    user_model = {**PHELPS_DEATON, "n_periods": 1}
    model = process_model(user_model)

    _, space_info, _, _ = create_state_choice_space(
        model=model,
        period=0,
        is_last_period=False,
        jit_filter=False,
    )

    next_state = get_next_state_function(model)
    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        data_name="vf_arr",
        interpolation_options={},
        is_last_period=True,
    )

    compute_ccv_policy_functions = model.n_periods * [
        create_compute_conditional_continuation_policy(
            utility_and_feasibility=u_and_f,
            continuous_choice_variables=["consumption"],
        ),
    ]

    return {
        "state_indexers": [{}],
        "continuous_choice_grids": [
            {"consumption": jnp.linspace(1, 100, num=N_CHOICE_GRID_POINTS)},
        ],
        "compute_ccv_policy_functions": compute_ccv_policy_functions,
        "model": model,
        "next_state": next_state,
    }


def test_simulate_using_raw_inputs(simulate_inputs):
    params = {
        "beta": 1.0,
        "utility": {"delta": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    got = simulate(
        params=params,
        vf_arr_list=[None],
        initial_states={"wealth": jnp.array([1.0, 50.400803])},
        **simulate_inputs,
    )

    assert_array_equal(got.loc[0, :]["retirement"], 1)
    assert_array_almost_equal(got.loc[0, :]["consumption"], jnp.array([1.0, 50.400803]))


# ======================================================================================
# Test simulate using get_lcm_function
# ======================================================================================


@pytest.fixture()
def phelps_deaton_model_solution():
    def _model_solution(n_periods):
        model = {**PHELPS_DEATON, "n_periods": n_periods}
        solve_model, _ = get_lcm_function(model=model)

        params = {
            "beta": 1.0,
            "utility": {"delta": 1.0},
            "next_wealth": {
                "interest_rate": 0.05,
                "wage": 1.0,
            },
        }

        vf_arr_list = solve_model(params)
        return vf_arr_list, params, model

    return _model_solution


@pytest.mark.parametrize("n_periods", range(1, PHELPS_DEATON["n_periods"] + 1))
def test_simulate_using_get_lcm_function(phelps_deaton_model_solution, n_periods):
    vf_arr_list, params, model = phelps_deaton_model_solution(n_periods)

    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

    res = simulate_model(
        params,
        vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([1.0, 20, 40, 70]),
        },
        targets=["utility", "working"],
    )

    # assert that everyone retires in the last period
    last_period_index = n_periods - 1
    assert_array_equal(res.loc[last_period_index, :]["retirement"], 1)

    # assert that higher wealth leads to higher consumption
    for period in range(n_periods):
        assert (res.loc[period, :]["consumption"].diff()[1:] >= 0).all()

        # The following does not work. I.e. the continuation value in each period is not
        # weakly increasing in wealth. It is unclear if this needs to hold.
        # ------------------------------------------------------------------------------
        # assert jnp.all(jnp.diff(res[period]["value"]) >= 0)  # noqa: ERA001


# ======================================================================================
# Testing effects of parameters
# ======================================================================================


def test_effect_of_beta_on_last_period():
    model = {**PHELPS_DEATON, "n_periods": 5}

    # Model solutions
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model, targets="solve")

    params = {
        "beta": None,
        "utility": {"delta": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    # low beta
    params_low = params.copy()
    params_low["beta"] = 0.5

    # high delta
    params_high = params.copy()
    params_high["beta"] = 0.99

    # solutions
    solution_low = solve_model(params_low)
    solution_high = solve_model(params_high)

    # Simulate
    # ==================================================================================
    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

    initial_wealth = jnp.array([20.0, 50, 70])

    res_low = simulate_model(
        params_low,
        vf_arr_list=solution_low,
        initial_states={"wealth": initial_wealth},
    )

    res_high = simulate_model(
        params_high,
        vf_arr_list=solution_high,
        initial_states={"wealth": initial_wealth},
    )

    # Asserting
    # ==================================================================================
    last_period_index = 4
    assert (
        res_low.loc[last_period_index, :]["value"]
        <= res_high.loc[last_period_index, :]["value"]
    ).all()


def test_effect_of_delta():
    model = {**PHELPS_DEATON, "n_periods": 5}

    # Model solutions
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model, targets="solve")

    params = {
        "beta": 1.0,
        "utility": {"delta": None},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    # low delta
    params_low = params.copy()
    params_low["utility"]["delta"] = 0.2

    # high delta
    params_high = params.copy()
    params_high["utility"]["delta"] = 1.5

    # solutions
    solution_low = solve_model(params_low)
    solution_high = solve_model(params_high)

    # Simulate
    # ==================================================================================
    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

    initial_wealth = jnp.array([20.0, 50, 70])

    res_low = simulate_model(
        params_low,
        vf_arr_list=solution_low,
        initial_states={"wealth": initial_wealth},
    )

    res_high = simulate_model(
        params_high,
        vf_arr_list=solution_high,
        initial_states={"wealth": initial_wealth},
    )

    # Asserting
    # ==================================================================================
    for period in range(5):
        assert (
            res_low.loc[period, :]["consumption"]
            <= res_high.loc[period, :]["consumption"]
        ).all()
        assert (
            res_low.loc[period, :]["retirement"]
            >= res_high.loc[period, :]["retirement"]
        ).all()


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


def test_filter_ccv_policy():
    ccc_policy = jnp.array(
        [
            [0, 1],
            [1, 0],
        ],
    )
    dense_argmax = jnp.array([0, 1])
    dense_vars_grid_shape = (2,)
    got = filter_ccv_policy(
        ccv_policy=ccc_policy,
        dense_argmax=dense_argmax,
        dense_vars_grid_shape=dense_vars_grid_shape,
    )
    assert jnp.all(got == jnp.array([0, 0]))


def test_create_data_state_choice_space():
    model = process_model(PHELPS_DEATON_WITH_FILTERS)
    got_space, got_segment_info = create_data_scs(
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


def test_determine_discrete_dense_choice_axes():
    variable_info = pd.DataFrame(
        {
            "is_state": [True, True, False, True, False, False],
            "is_dense": [False, True, True, False, True, True],
            "is_choice": [False, False, True, True, True, True],
            "is_continuous": [False, True, False, False, False, True],
        },
    )
    got = determine_discrete_dense_choice_axes(variable_info)
    assert got == (1, 2)
