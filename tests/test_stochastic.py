import jax.numpy as jnp
import lcm
import pandas as pd
import pytest
from lcm.entry_point import (
    get_lcm_function,
)

from tests.test_models.stochastic import MODEL_CONFIG, PARAMS

# ======================================================================================
# Simulate
# ======================================================================================


def test_get_lcm_function_with_simulate_target():
    simulate_model, _ = get_lcm_function(
        model=MODEL_CONFIG,
        targets="solve_and_simulate",
    )

    res = simulate_model(
        PARAMS,
        initial_states={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        },
    )

    expected_partner = [
        0,
        0,
        1,
        0,  # period 0
        1,
        1,
        1,
        1,  # period 1
        1,
        1,
        1,
        0,  # period 2
    ]
    assert jnp.array_equal(res["partner"].values, expected_partner)


# ======================================================================================
# Solve
# ======================================================================================


def test_get_lcm_function_with_solve_target():
    solve_model, _ = get_lcm_function(model=MODEL_CONFIG)
    solve_model(PARAMS)


# ======================================================================================
# Comparison with deterministic results
# ======================================================================================


@pytest.fixture()
def model_and_params():
    def utility(consumption, working, health, delta, gamma):
        return jnp.log(consumption) + (gamma * health - delta) * working

    def next_wealth(wealth, consumption, working, wage, interest_rate):
        return (1 + interest_rate) * (wealth - consumption) + wage * working

    @lcm.mark.stochastic
    def next_health_stochastic(health):  # noqa: ARG001
        pass

    def next_health_deterministic(health):
        return health

    def consumption_constraint(consumption, wealth):
        return consumption <= wealth

    _model = {
        "functions": {
            "utility": utility,
            "next_wealth": next_wealth,
            "consumption_constraint": consumption_constraint,
        },
        "choices": {
            "working": {"options": [0, 1]},
            "consumption": {
                "grid_type": "linspace",
                "start": 0,
                "stop": 100,
                "n_points": 50,
            },
        },
        "states": {
            "health": {"options": [0, 1]},
            "wealth": {
                "grid_type": "linspace",
                "start": 0,
                "stop": 100,
                "n_points": 10,
            },
        },
        "n_periods": 3,
    }

    model_deterministic = _model.copy()
    model_deterministic["functions"]["next_health"] = next_health_deterministic

    model_stochastic = _model.copy()
    model_stochastic["functions"]["next_health"] = next_health_stochastic

    params = {
        "beta": 0.95,
        "utility": {"delta": 0.5, "gamma": 0.5},
        "next_wealth": {"interest_rate": 0.05, "wage": 10.0},
        "next_health": {},
        "consumption_constraint": {},
        "shocks": {
            "health": jnp.identity(2),
        },
    }

    return model_deterministic, model_stochastic, params


def test_compare_deterministic_and_stochastic_results(model_and_params):
    """Test that the deterministic and stochastic models produce the same results."""
    model_deterministic, model_stochastic, params = model_and_params

    # ==================================================================================
    # Compare value function arrays
    # ==================================================================================
    solve_model_deterministic, _ = get_lcm_function(model=model_deterministic)
    solve_model_stochastic, _ = get_lcm_function(model=model_stochastic)

    solution_deterministic = solve_model_deterministic(params)
    solution_stochastic = solve_model_stochastic(params)

    assert jnp.array_equal(solution_deterministic, solution_stochastic, equal_nan=True)

    # ==================================================================================
    # Compare simulation results
    # ==================================================================================
    simulate_model_deterministic, _ = get_lcm_function(
        model=model_deterministic,
        targets="simulate",
    )
    simulate_model_stochastic, _ = get_lcm_function(
        model=model_stochastic,
        targets="simulate",
    )

    initial_states = {
        "health": jnp.array([1, 1, 0, 0]),
        "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
    }

    simulation_deterministic = simulate_model_deterministic(
        params,
        vf_arr_list=solution_deterministic,
        initial_states=initial_states,
    )
    simulation_stochastic = simulate_model_stochastic(
        params,
        vf_arr_list=solution_stochastic,
        initial_states=initial_states,
    )
    pd.testing.assert_frame_equal(simulation_deterministic, simulation_stochastic)
