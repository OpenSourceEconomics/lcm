import jax.numpy as jnp
import pytest
from pybaum import tree_equal, tree_map

from lcm.entry_point import (
    create_compute_conditional_continuation_policy,
    create_compute_conditional_continuation_value,
    get_lcm_function,
)
from lcm.input_processing import process_model
from lcm.model_functions import get_utility_and_feasibility_function
from lcm.state_space import create_state_choice_space
from tests.test_models import get_model_config
from tests.test_models.deterministic import RetirementStatus
from tests.test_models.deterministic import utility as iskhakov_et_al_2017_utility
from tests.test_models.discrete_deterministic import ConsumptionChoice

# ======================================================================================
# Test cases
# ======================================================================================


STRIPPED_DOWN_AND_DISCRETE_MODELS = [
    "iskhakov_et_al_2017_stripped_down",
    "iskhakov_et_al_2017_discrete",
]


# ======================================================================================
# Solve
# ======================================================================================


def test_get_lcm_function_with_solve_target_stripped_down():
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    solve_model, params_template = get_lcm_function(model=model)

    params = tree_map(lambda _: 0.2, params_template)

    solve_model(params)


def test_get_lcm_function_with_solve_target_fully_discrete():
    model = get_model_config("iskhakov_et_al_2017_discrete", n_periods=3)
    solve_model, params_template = get_lcm_function(model=model)

    params = tree_map(lambda _: 0.2, params_template)

    solve_model(params)


# ======================================================================================
# Simulate
# ======================================================================================


def test_get_lcm_function_with_simulation_target_simple_stripped_down():
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)

    simulate, params_template = get_lcm_function(
        model=model,
        targets="solve_and_simulate",
    )
    params = tree_map(lambda _: 0.2, params_template)

    simulate(
        params,
        initial_states={
            "wealth": jnp.array([0.0, 10.0, 50.0]),
        },
        additional_targets=["age"] if "age" in model.functions else None,
    )


def test_get_lcm_function_with_simulation_target_simple_fully_discrete():
    model = get_model_config("iskhakov_et_al_2017_discrete", n_periods=3)

    simulate, params_template = get_lcm_function(
        model=model,
        targets="solve_and_simulate",
    )
    params = tree_map(lambda _: 0.2, params_template)

    simulate(
        params,
        initial_states={
            "wealth": jnp.array([0.0, 10.0, 50.0]),
        },
        additional_targets=["age"] if "age" in model.functions else None,
    )


@pytest.mark.parametrize(
    "model",
    [get_model_config(name, n_periods=3) for name in STRIPPED_DOWN_AND_DISCRETE_MODELS],
    ids=STRIPPED_DOWN_AND_DISCRETE_MODELS,
)
def test_get_lcm_function_with_simulation_is_coherent(model):
    """Test that solve_and_simulate creates same output as solve then simulate."""
    # solve then simulate
    # ==================================================================================

    # solve
    solve_model, params_template = get_lcm_function(model=model)
    params = tree_map(lambda _: 0.2, params_template)
    vf_arr_list = solve_model(params)

    # simulate using solution
    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

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
        model=model,
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
    "model",
    [get_model_config("iskhakov_et_al_2017", n_periods=3)],
    ids=["iskhakov_et_al_2017"],
)
def test_get_lcm_function_with_simulation_target_iskhakov_et_al_2017(model):
    # solve model
    solve_model, params_template = get_lcm_function(model=model, targets="solve")
    params = tree_map(lambda _: 0.2, params_template)
    vf_arr_list = solve_model(params)

    # simulate using solution
    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

    simulate_model(
        params,
        vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([10.0, 10.0, 20.0]),
            "lagged_retirement": jnp.array(
                [
                    RetirementStatus.working,
                    RetirementStatus.retired,
                    RetirementStatus.retired,
                ]
            ),
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

    _, space_info = create_state_choice_space(
        model=model,
        is_last_period=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv = create_compute_conditional_continuation_value(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=["consumption"],
    )

    val = compute_ccv(
        consumption=jnp.array([10, 20, 30.0]),
        retirement=RetirementStatus.retired,
        wealth=30,
        params=params,
        vf_arr=None,
    )
    assert val == iskhakov_et_al_2017_utility(
        consumption=30.0,
        working=RetirementStatus.working,
        disutility_of_work=1.0,
    )


def test_create_compute_conditional_continuation_value_with_discrete_model():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_discrete", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    _, space_info = create_state_choice_space(
        model=model,
        is_last_period=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv = create_compute_conditional_continuation_value(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=[],
    )

    val = compute_ccv(
        consumption=jnp.array([ConsumptionChoice.low, ConsumptionChoice.high]),
        retirement=RetirementStatus.retired,
        wealth=2,
        params=params,
        vf_arr=None,
    )
    assert val == iskhakov_et_al_2017_utility(
        consumption=2,
        working=RetirementStatus.working,
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

    _, space_info = create_state_choice_space(
        model=model,
        is_last_period=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv_policy = create_compute_conditional_continuation_policy(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=["consumption"],
    )

    policy, val = compute_ccv_policy(
        consumption=jnp.array([10, 20, 30.0]),
        retirement=RetirementStatus.retired,
        wealth=30,
        params=params,
        vf_arr=None,
    )
    assert policy == 2
    assert val == iskhakov_et_al_2017_utility(
        consumption=30.0,
        working=RetirementStatus.working,
        disutility_of_work=1.0,
    )


def test_create_compute_conditional_continuation_policy_with_discrete_model():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_discrete", n_periods=3),
    )

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    _, space_info = create_state_choice_space(
        model=model,
        is_last_period=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        period=model.n_periods - 1,
        is_last_period=True,
    )

    compute_ccv_policy = create_compute_conditional_continuation_policy(
        utility_and_feasibility=u_and_f,
        continuous_choice_variables=[],
    )

    policy, val = compute_ccv_policy(
        consumption=jnp.array([ConsumptionChoice.low, ConsumptionChoice.high]),
        retirement=RetirementStatus.retired,
        wealth=2,
        params=params,
        vf_arr=None,
    )
    assert policy == 1
    assert val == iskhakov_et_al_2017_utility(
        consumption=2,
        working=RetirementStatus.working,
        disutility_of_work=1.0,
    )


# ======================================================================================
# Test constraints with _period argument
# ======================================================================================


def test_get_lcm_function_with_period_argument_in_constraint():
    model = get_model_config("iskhakov_et_al_2017", n_periods=3)

    def absorbing_retirement_constraint(retirement, lagged_retirement, _period):
        return jnp.logical_or(
            retirement == RetirementStatus.retired,
            lagged_retirement == RetirementStatus.working,
        )

    model.functions["absorbing_retirement_constraint"] = absorbing_retirement_constraint

    solve_model, params_template = get_lcm_function(model=model)
    params = tree_map(lambda _: 0.2, params_template)
    solve_model(params)
