from copy import deepcopy

import jax.numpy as jnp
import pytest
from lcm.entry_point import get_lcm_function


def utility(consumption, working, wealth):  # noqa: ARG001
    return jnp.log(consumption)


def next_wealth(wealth, consumption, working):
    return wealth - consumption + working


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


base_model = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
    },
    "choices": {
        "consumption": {"options": [1, 2]},
        "working": {"options": [0, 1]},
    },
    "states": {},
    "n_periods": 2,
}


model_with_cont_state = deepcopy(base_model)
model_with_cont_state["states"]["wealth"] = {
    "grid_type": "linspace",
    "start": 1,
    "stop": 2,
    "n_points": 2,
}


model_with_discrete_state = deepcopy(base_model)
model_with_discrete_state["states"]["wealth"] = {"options": [1, 2]}

MODELS = {"continuous": model_with_cont_state, "discrete": model_with_discrete_state}


@pytest.mark.parametrize("model", ["continuous", "discrete"])
def test_model(model):
    _model = MODELS[model]

    solve_and_simulate, _ = get_lcm_function(
        model=_model,
        targets="solve_and_simulate",
    )

    solve_and_simulate(
        params={"beta": 1.0},
        initial_states={
            "wealth": jnp.array([1.0, 2.0]),
        },
    )
