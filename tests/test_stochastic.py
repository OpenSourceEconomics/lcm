import jax.numpy as jnp
from lcm.entry_point import (
    get_lcm_function,
)
from lcm.example_models_stochastic import MODEL, PARAMS

# ======================================================================================
# Simulate
# ======================================================================================


def test_get_lcm_function_with_simulate_target():
    simulate_model, _ = get_lcm_function(model=MODEL, targets="solve_and_simulate")

    simulate_model(
        PARAMS,
        initial_states={
            "health": jnp.array([1, 1, 0, 0]),
            "partner": jnp.array([0, 0, 1, 0]),
            "wealth": jnp.array([10.0, 50.0, 30, 80.0]),
        },
    )


# ======================================================================================
# Solve
# ======================================================================================


def test_get_lcm_function_with_solve_target():
    solve_model, _ = get_lcm_function(model=MODEL)
    solve_model(PARAMS)
