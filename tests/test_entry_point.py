import jax.numpy as jnp
import pytest
from lcm.entry_point import (
    create_compute_conditional_continuation_policy,
    create_compute_conditional_continuation_value,
    get_lcm_function,
)
from lcm.model_functions import get_utility_and_feasibility_function
from lcm.process_model import process_model
from lcm.state_space import create_state_choice_space
from pybaum import tree_equal, tree_map

from tests.test_models.deterministic import get_model_config
from tests.test_models.deterministic import utility as iskhakov_et_al_2017_utility

# ======================================================================================
# Test cases
# ======================================================================================


STRIPPED_DOWN_AND_FULLY_DISCRETE_MODELS = [
    "iskhakov_et_al_2017_stripped_down",
    "iskhakov_et_al_2017_fully_discrete",
]


# ======================================================================================
# Solve
# ======================================================================================


@pytest.mark.parametrize(
    "user_model",
    [
        get_model_config(name, n_periods=3)
        for name in ["iskhakov_et_al_2017", *STRIPPED_DOWN_AND_FULLY_DISCRETE_MODELS]
    ],
    ids=["iskhakov_et_al_2017", *STRIPPED_DOWN_AND_FULLY_DISCRETE_MODELS],
)
def test_get_lcm_function_with_solve_target(user_model):
    solve_model, params_template = get_lcm_function(model=user_model)

    params = tree_map(lambda _: 0.2, params_template)

    solve_model(params)


# ======================================================================================
# Simulate
# ======================================================================================


@pytest.mark.parametrize(
    "user_model",
    [
        get_model_config(name, n_periods=3)
        for name in STRIPPED_DOWN_AND_FULLY_DISCRETE_MODELS
    ],
    ids=STRIPPED_DOWN_AND_FULLY_DISCRETE_MODELS,
)
def test_get_lcm_function_with_simulation_target_simple(user_model):
    simulate, params_template = get_lcm_function(
        model=user_model,
        targets="solve_and_simulate",
    )
    params = tree_map(lambda _: 0.2, params_template)

    simulate(
        params,
        initial_states={
            "wealth": jnp.array([0.0, 10.0, 50.0]),
        },
        additional_targets=["age"] if "age" in user_model["functions"] else None,
    )


@pytest.mark.parametrize(
    "user_model",
    [
        get_model_config(name, n_periods=3)
        for name in STRIPPED_DOWN_AND_FULLY_DISCRETE_MODELS
    ],
    ids=STRIPPED_DOWN_AND_FULLY_DISCRETE_MODELS,
)
def test_get_lcm_function_with_simulation_is_coherent(user_model):
    """Test that solve_and_simulate creates same output as solve then simulate."""
    # solve then simulate
    # ==================================================================================

    # solve
    solve_model, params_template = get_lcm_function(model=user_model)
    params = tree_map(lambda _: 0.2, params_template)
    vf_arr_list = solve_model(params)

    # simulate using solution
    simulate_model, _ = get_lcm_function(model=user_model, targets="simulate")

    solve_then_simulate = simulate_model(
        params,
        vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([0.0, 10.0, 50.0]),
        },
    )

    # solve and simulate
    # ==================================================================================
    solve_and_simulate_model, _ = get_lcm_function(
        model=user_model,
        targets="solve_and_simulate",
    )

    solve_and_simulate = solve_and_simulate_model(
        params,
        initial_states={
            "wealth": jnp.array([0.0, 10.0, 50.0]),
        },
    )

    assert tree_equal(solve_then_simulate, solve_and_simulate)


@pytest.mark.parametrize(
    "user_model",
    [get_model_config("iskhakov_et_al_2017", n_periods=3)],
    ids=["iskhakov_et_al_2017"],
)
def test_get_lcm_function_with_simulation_target_iskhakov_et_al_2017(user_model):
    # solve model
    solve_model, params_template = get_lcm_function(model=user_model, targets="solve")
    params = tree_map(lambda _: 0.2, params_template)
    vf_arr_list = solve_model(params)

    # simulate using solution
    simulate_model, _ = get_lcm_function(model=user_model, targets="simulate")

    simulate_model(
        params,
        vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([10.0, 10.0, 20.0]),
            "lagged_retirement": jnp.array([0, 1, 1]),
        },
    )


# ======================================================================================
# Create compute conditional continuation value
# ======================================================================================


def test_create_compute_conditional_continuation_value():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    _, space_info, _, _ = create_state_choice_space(
        model=model,
        period=0,
        is_last_period=False,
        jit_filter=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        data_name="vf_arr",
        interpolation_options={},
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv = create_compute_conditional_continuation_value(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=["consumption"],
    )

    val = compute_ccv(
        consumption=jnp.array([10, 20, 30.0]),
        retirement=1,
        wealth=30,
        params=params,
        vf_arr=None,
    )
    assert val == iskhakov_et_al_2017_utility(
        consumption=30.0,
        working=0,
        disutility_of_work=1.0,
    )


def test_create_compute_conditional_continuation_value_with_discrete_model():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_fully_discrete", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    _, space_info, _, _ = create_state_choice_space(
        model=model,
        period=0,
        is_last_period=False,
        jit_filter=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        data_name="vf_arr",
        interpolation_options={},
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv = create_compute_conditional_continuation_value(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=[],
    )

    val = compute_ccv(
        consumption_index=jnp.array([0, 1]),
        retirement=1,
        wealth=2,
        params=params,
        vf_arr=None,
    )
    assert val == iskhakov_et_al_2017_utility(
        consumption=2,
        working=0,
        disutility_of_work=1.0,
    )


# ======================================================================================
# Create compute conditional continuation policy
# ======================================================================================


def test_create_compute_conditional_continuation_policy():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    _, space_info, _, _ = create_state_choice_space(
        model=model,
        period=0,
        is_last_period=False,
        jit_filter=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        data_name="vf_arr",
        interpolation_options={},
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv_policy = create_compute_conditional_continuation_policy(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=["consumption"],
    )

    policy, val = compute_ccv_policy(
        consumption=jnp.array([10, 20, 30.0]),
        retirement=1,
        wealth=30,
        params=params,
        vf_arr=None,
    )
    assert policy == 2
    assert val == iskhakov_et_al_2017_utility(
        consumption=30.0,
        working=0,
        disutility_of_work=1.0,
    )


def test_create_compute_conditional_continuation_policy_with_discrete_model():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_fully_discrete", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    _, space_info, _, _ = create_state_choice_space(
        model=model,
        period=0,
        is_last_period=False,
        jit_filter=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        data_name="vf_arr",
        interpolation_options={},
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv_policy = create_compute_conditional_continuation_policy(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=[],
    )

    policy, val = compute_ccv_policy(
        consumption_index=jnp.array([0, 1]),
        retirement=1,
        wealth=2,
        params=params,
        vf_arr=None,
    )
    assert policy == 1
    assert val == iskhakov_et_al_2017_utility(
        consumption=2,
        working=0,
        disutility_of_work=1.0,
    )


# ======================================================================================
# Test filter with _period argument
# ======================================================================================


def test_get_lcm_function_with_period_argument_in_filter():
    user_model = get_model_config("iskhakov_et_al_2017", n_periods=3)

    def absorbing_retirement_filter(retirement, lagged_retirement, _period):
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    user_model["functions"]["absorbing_retirement_filter"] = absorbing_retirement_filter

    solve_model, params_template = get_lcm_function(model=user_model)
    params = tree_map(lambda _: 0.2, params_template)
    solve_model(params)
