import jax.numpy as jnp
import pandas as pd
from jax import Array
from pybaum import tree_equal

from lcm.input_processing import process_model
from lcm.interfaces import InternalModel
from lcm.next_state import _get_stochastic_next_func, get_next_state_function
from lcm.typing import ParamsDict, Scalar, ShockType, Target
from tests.test_models import get_model_config

# ======================================================================================
# Solve target
# ======================================================================================


def test_get_next_state_function_with_solve_target():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )
    got_func = get_next_state_function(model, target=Target.SOLVE)

    params = {
        "beta": 1.0,
        "utility": {"disutility_of_work": 1.0},
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
    def f_a(state: Array, params: ParamsDict) -> Scalar:  # noqa: ARG001
        return state[0]

    def f_b(state: Scalar, params: ParamsDict) -> Scalar:  # noqa: ARG001
        return None  # type: ignore[return-value]

    def f_weight_b(state: Scalar, params: ParamsDict) -> Array:  # noqa: ARG001
        return jnp.array([[0.0, 1.0]])

    functions = {
        "a": f_a,
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

    model = InternalModel(
        functions=functions,  # type: ignore[arg-type]
        grids=grids,
        function_info=function_info,
        gridspecs={},
        variable_info=pd.DataFrame(),
        params={},
        random_utility_shocks=ShockType.NONE,
        n_periods=1,
    )

    got_func = get_next_state_function(model, target=Target.SOLVE)

    keys = {"b": jnp.arange(2, dtype="uint32")}
    got = got_func(state=jnp.arange(2), keys=keys)

    expected = {"a": jnp.array([0]), "b": jnp.array([1])}
    assert tree_equal(expected, got)


def test_get_stochastic_next_func():
    grids = {"a": jnp.arange(2)}
    got_func = _get_stochastic_next_func(name="a", grids=grids)

    keys = {"a": jnp.arange(2, dtype="uint32")}  # PRNG dtype
    weights = jnp.array([[0.0, 1], [1, 0]])
    got = got_func(keys=keys, weight_a=weights)  # type: ignore[call-arg]

    assert jnp.array_equal(got, jnp.array([1, 0]))
