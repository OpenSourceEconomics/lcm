import jax.numpy as jnp
import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON, PHELPS_DEATON_WITH_FILTERS
from pybaum import tree_map

MODELS = {
    "simple": PHELPS_DEATON,
    "with_filters": PHELPS_DEATON_WITH_FILTERS,
}


@pytest.mark.parametrize("user_model", list(MODELS.values()), ids=list(MODELS))
def test_get_lcm_function_with_solve_target(user_model):
    solve_model, params_template = get_lcm_function(model=user_model)

    params = tree_map(lambda _: 0.2, params_template)

    solve_model(params)


# ======================================================================================
# Simulate - Test functionality
# ======================================================================================


@pytest.mark.parametrize("user_model", [PHELPS_DEATON], ids=["simple"])
def test_get_lcm_function_with_simulation_target_simple(user_model):
    # solve model
    solve_model, params_template = get_lcm_function(model=user_model)
    params = tree_map(lambda _: 0.2, params_template)
    vf_arr_list = solve_model(params)

    # simulate using solution
    simulate_model, _ = get_lcm_function(model=user_model, targets="simulate")

    simulate_model(
        params,
        vf_arr_list=vf_arr_list,
        initial_states={
            "wealth": jnp.array([0.0, 10.0, 50.0]),
        },
    )


@pytest.mark.parametrize(
    "user_model",
    [PHELPS_DEATON_WITH_FILTERS],
    ids=["with_filters"],
)
def test_get_lcm_function_with_simulation_target_with_filters(user_model):
    # solve model
    solve_model, params_template = get_lcm_function(model=user_model)
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
# Simulate - Test correctness
# ======================================================================================


def test_get_lcm_function_with_simulation_three_periods():
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

    # simulate using solution
    simulate_model, _ = get_lcm_function(model=user_model, targets="simulate")

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
