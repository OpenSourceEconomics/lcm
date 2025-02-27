import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from lcm.conditional_continuation import (
    get_compute_conditional_continuation_policy,
)
from lcm.entry_point import get_lcm_function
from lcm.input_processing import process_model
from lcm.logging import get_logger
from lcm.next_state import get_next_state_function
from lcm.simulation.simulate import (
    determine_discrete_choice_axes,
    filter_ccv_policy,
    retrieve_choices,
    simulate,
)
from lcm.solution.state_choice_space import create_state_choice_space
from lcm.typing import Target
from lcm.utility_and_feasibility import get_utility_and_feasibility_function
from tests.test_models import (
    get_model_config,
    get_params,
)

# ======================================================================================
# Test simulate using raw inputs
# ======================================================================================


@pytest.fixture
def simulate_inputs():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=1)
    model = process_model(model_config)

    state_space_info = create_state_choice_space(
        model=model,
        is_last_period=False,
    )[1]

    compute_ccv_policy_functions = []
    for period in range(model.n_periods):
        u_and_f = get_utility_and_feasibility_function(
            model=model,
            next_state_space_info=state_space_info,
            period=period,
            is_last_period=True,
        )
        compute_ccv = get_compute_conditional_continuation_policy(
            utility_and_feasibility=u_and_f,
            continuous_choice_variables=("consumption",),
        )
        compute_ccv_policy_functions.append(compute_ccv)

    n_grid_points = model_config.choices["consumption"].n_points  # type: ignore[attr-defined]

    return {
        "continuous_choice_grids": [
            {"consumption": jnp.linspace(1, 100, num=n_grid_points)},
        ],
        "compute_ccv_policy_functions": compute_ccv_policy_functions,
        "model": model,
        "next_state": get_next_state_function(model, target=Target.SIMULATE),
    }


def test_simulate_using_raw_inputs(simulate_inputs):
    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
        },
    }

    got = simulate(
        params=params,
        pre_computed_vf_arr_list=[jnp.empty(0)],
        initial_states={"wealth": jnp.array([1.0, 50.400803])},
        logger=get_logger(debug_mode=False),
        **simulate_inputs,
    )

    assert_array_equal(got.loc[0, :]["retirement"], 1)
    assert_array_almost_equal(got.loc[0, :]["consumption"], jnp.array([1.0, 50.400803]))


# ======================================================================================
# Test simulate using get_lcm_function
# ======================================================================================


@pytest.fixture
def iskhakov_et_al_2017_stripped_down_model_solution():
    def _model_solution(n_periods):
        model_config = get_model_config(
            "iskhakov_et_al_2017_stripped_down",
            n_periods=n_periods,
        )
        updated_functions = {
            # remove dependency on age, so that wage becomes a parameter
            name: func
            for name, func in model_config.functions.items()
            if name not in ["age", "wage"]
        }
        model_config = model_config.replace(functions=updated_functions)
        solve_model, _ = get_lcm_function(model_config, targets="solve")

        params = get_params()
        vf_arr_list = solve_model(params=params)
        return vf_arr_list, params, model_config

    return _model_solution


def test_simulate_using_get_lcm_function(
    iskhakov_et_al_2017_stripped_down_model_solution,
):
    n_periods = 3
    vf_arr_list, params, model = iskhakov_et_al_2017_stripped_down_model_solution(
        n_periods=n_periods,
    )

    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

    res: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params,
        pre_computed_vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([20.0, 150, 250, 320]),
        },
        additional_targets=["utility", "consumption_constraint"],
    )

    assert {
        "_period",
        "value",
        "retirement",
        "consumption",
        "wealth",
        "utility",
        "consumption_constraint",
    } == set(res.columns)

    # assert that everyone retires in the last period
    last_period_index = n_periods - 1
    assert_array_equal(res.loc[last_period_index, :]["retirement"], 1)

    for period in range(n_periods):
        # assert that higher wealth leads to higher consumption in each period
        assert (res.loc[period]["consumption"].diff()[1:] >= 0).all()  # type: ignore[operator]

        # assert that higher wealth leads to higher value function in each period
        assert (res.loc[period]["value"].diff()[1:] >= 0).all()  # type: ignore[operator]


def test_simulate_with_only_discrete_choices():
    model = get_model_config("iskhakov_et_al_2017_discrete", n_periods=2)
    params = get_params(wage=1.5, beta=1, interest_rate=0)

    simulate_model, _ = get_lcm_function(model=model, targets="solve_and_simulate")

    res: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params,
        initial_states={"wealth": jnp.array([0, 4])},
        additional_targets=["labor_income", "working"],
    )

    assert_array_equal(res["retirement"], jnp.array([0, 1, 1, 1]))
    assert_array_equal(res["consumption"], jnp.array([0, 1, 1, 1]))
    assert_array_equal(res["wealth"], jnp.array([0, 4, 2, 2]))


# ======================================================================================
# Testing effects of parameters
# ======================================================================================


def test_effect_of_beta_on_last_period():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=5)

    # Model solutions
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model_config, targets="solve")

    # low beta
    params_low = get_params(beta=0.5, disutility_of_work=1.0)

    # high beta
    params_high = get_params(beta=0.99, disutility_of_work=1.0)

    # solutions
    solution_low = solve_model(params_low)
    solution_high = solve_model(params_high)

    # Simulate
    # ==================================================================================
    simulate_model, _ = get_lcm_function(model=model_config, targets="simulate")

    initial_wealth = jnp.array([20.0, 50, 70])

    res_low: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_low,
        pre_computed_vf_arr_list=solution_low,
        initial_states={"wealth": initial_wealth},
    )

    res_high: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_high,
        pre_computed_vf_arr_list=solution_high,
        initial_states={"wealth": initial_wealth},
    )

    # Asserting
    # ==================================================================================
    last_period_index = 4
    assert (
        res_low.loc[last_period_index, :]["value"]
        <= res_high.loc[last_period_index, :]["value"]
    ).all()


def test_effect_of_disutility_of_work():
    model_config = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=5)

    # Model solutions
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model_config, targets="solve")

    # low disutility_of_work
    params_low = get_params(beta=1.0, disutility_of_work=0.2)

    # high disutility_of_work
    params_high = get_params(beta=1.0, disutility_of_work=1.5)

    # solutions
    solution_low = solve_model(params_low)
    solution_high = solve_model(params_high)

    # Simulate
    # ==================================================================================
    simulate_model, _ = get_lcm_function(model=model_config, targets="simulate")

    initial_wealth = jnp.array([20.0, 50, 70])

    res_low: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_low,
        pre_computed_vf_arr_list=solution_low,
        initial_states={"wealth": initial_wealth},
    )

    res_high: pd.DataFrame = simulate_model(  # type: ignore[assignment]
        params_high,
        pre_computed_vf_arr_list=solution_high,
        initial_states={"wealth": initial_wealth},
    )

    # Asserting
    # ==================================================================================
    for period in range(5):
        # We expect that individuals with lower disutility of work, work (weakly) more
        # and thus consume (weakly) more
        assert (
            res_low.loc[period]["consumption"] >= res_high.loc[period]["consumption"]
        ).all()

        # We expect that individuals with lower disutility of work retire (weakly) later
        assert (
            res_low.loc[period]["retirement"] <= res_high.loc[period]["retirement"]
        ).all()


# ======================================================================================
# Helper functions
# ======================================================================================


def test_retrieve_choices():
    got = retrieve_choices(
        flat_indices=jnp.array([0, 3, 7]),
        grids={"a": jnp.linspace(0, 1, 5), "b": jnp.linspace(10, 20, 6)},
        grids_shapes=(5, 6),
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
    argmax = jnp.array([0, 1])
    vars_grid_shape = (2,)
    got = filter_ccv_policy(
        ccv_policy=ccc_policy,
        discrete_argmax=argmax,
        vars_grid_shape=vars_grid_shape,
    )
    assert jnp.all(got == jnp.array([0, 0]))


def test_determine_discrete_choice_axes():
    variable_info = pd.DataFrame(
        {
            "is_state": [True, True, False, True, False, False],
            "is_choice": [False, False, True, True, True, True],
            "is_discrete": [True, True, True, True, True, False],
            "is_continuous": [False, True, False, False, False, True],
        },
    )
    got = determine_discrete_choice_axes(variable_info)
    assert got == (1, 2, 3)
