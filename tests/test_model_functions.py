import jax.numpy as jnp
import pandas as pd
from numpy.testing import assert_array_equal

from lcm.input_processing import process_model
from lcm.interfaces import InternalModel
from lcm.model_functions import (
    get_combined_constraint,
    get_multiply_weights,
    get_utility_and_feasibility_function,
)
from lcm.state_space import create_state_choice_space
from tests.test_models import get_model_config
from tests.test_models.deterministic import utility


def test_get_combined_constraint():
    def f():
        return True

    def g():
        return False

    def h():
        return None

    function_info = pd.DataFrame(
        {"is_constraint": [True, True, False]},
        index=["f", "g", "h"],
    )
    model = InternalModel(
        grids=None,
        gridspecs=None,
        variable_info=None,
        functions={"f": f, "g": g, "h": h},
        function_info=function_info,
        params=None,
        random_utility_shocks=None,
        n_periods=None,
    )
    combined_constraint = get_combined_constraint(model)
    assert not combined_constraint()


def test_get_utility_and_feasibility_function():
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
        name_of_values_on_grid="vf_arr",
        period=model.n_periods - 1,
        is_last_period=True,
    )

    consumption = jnp.array([10, 20, 30])
    retirement = jnp.array([0, 1, 0])
    wealth = jnp.array([20, 20, 20])

    u, f = u_and_f(
        consumption=consumption,
        retirement=retirement,
        wealth=wealth,
        params=params,
        vf_arr=None,
    )

    assert_array_equal(
        u,
        utility(
            consumption=consumption,
            working=1 - retirement,
            disutility_of_work=1.0,
        ),
    )
    assert_array_equal(f, jnp.array([True, True, False]))


def test_get_multiply_weights():
    multiply_weights = get_multiply_weights(
        stochastic_variables=["a", "b"],
    )

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])

    got = multiply_weights(weight_next_a=a, weight_next_b=b)
    expected = jnp.array([[3, 4], [6, 8]])
    assert_array_equal(got, expected)
