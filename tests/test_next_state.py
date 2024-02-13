import jax.numpy as jnp
import pandas as pd
from lcm.example_models.basic_example_models import PHELPS_DEATON
from lcm.interfaces import Model
from lcm.next_state import _get_stochastic_next_func, get_next_state_function
from lcm.process_model import process_model
from pybaum import tree_equal

# ======================================================================================
# Solve target
# ======================================================================================


def test_get_next_state_function_with_solve_target():
    model = process_model(PHELPS_DEATON)
    got_func = get_next_state_function(model, target="solve")

    params = {
        "beta": 1.0,
        "utility": {"delta": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
        },
    }

    choice = {"retirement": 1, "consumption": 10}
    state = {"wealth": 20}

    got = got_func(**choice, **state, _period=1, params=params)
    assert got == {"next_wealth": 1.05 * (20 - 10)}


# ======================================================================================
# Simulate target
# ======================================================================================


def test_get_next_state_function_with_simulate_target():
    def f_b(state):  # noqa: ARG001
        return None

    def f_weight_b(state):  # noqa: ARG001
        return jnp.array([[0.0, 1.0]])

    functions = {
        "a": lambda state: state[0],
        "b": f_b,
        "weight_b": f_weight_b,
    }

    grids = {"b": jnp.arange(2)}

    function_info = pd.DataFrame(
        {
            "is_next": [True, True],
            "is_stochastic_next": [False, True],
        },
        index=["a", "b"],
    )

    model = Model(
        functions=functions,
        grids=grids,
        function_info=function_info,
        gridspecs=None,
        variable_info=None,
        params=None,
        shocks=None,
        n_periods=1,
    )

    got_func = get_next_state_function(model, target="solve")

    keys = {"b": jnp.arange(2, dtype="uint32")}
    got = got_func(state=jnp.arange(2), keys=keys)

    expected = {"a": jnp.array([0]), "b": jnp.array([1])}
    assert tree_equal(expected, got)


def test_get_stochastic_next_func():
    grids = {"a": jnp.arange(2)}
    got_func = _get_stochastic_next_func(name="a", grids=grids)

    keys = {"a": jnp.arange(2, dtype="uint32")}  # PRNG dtype
    weights = jnp.array([[0.0, 1], [1, 0]])
    got = got_func(keys=keys, weight_a=weights)

    assert jnp.array_equal(got, jnp.array([1, 0]))
