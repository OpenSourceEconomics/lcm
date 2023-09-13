import jax.numpy as jnp
from lcm.entry_point import (
    get_lcm_function,
)
from lcm.example_models_stochastic import MODEL

# ======================================================================================
# Solve
# ======================================================================================


def test_get_lcm_function_with_solve_target():
    solve_model, _ = get_lcm_function(model=MODEL)

    params = {
        "beta": 0.25,
        "utility": {"delta": 0.25, "gamma": 0.25},
        "next_wealth": {"interest_rate": 0.25, "wage": 0.25},
        "next_health": {},
        "consumption_constraint": {},
        "shocks": {"health": jnp.array([[0.25, 0.25], [0.25, 0.25]])},
    }

    solve_model(params)
