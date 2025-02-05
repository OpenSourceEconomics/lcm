import jax.numpy as jnp
import pandas as pd
import pytest
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


@pytest.mark.illustrative
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


@pytest.fixture
def internal_model_illustrative():
    def age(period):
        return period + 18

    def mandatory_retirement_constraint(retirement, age):
        # Individuals must be retired from age 65 onwards
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_constraint(lagged_retirement, age):
        # Individuals must have been retired last year from age 66 onwards
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_constraint(retirement, lagged_retirement):
        # If an individual was retired last year, it must be retired this year
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    functions = {
        "mandatory_retirement_constraint": mandatory_retirement_constraint,
        "mandatory_lagged_retirement_constraint": (
            mandatory_lagged_retirement_constraint
        ),
        "absorbing_retirement_constraint": absorbing_retirement_constraint,
        "age": age,
    }

    function_info = pd.DataFrame(
        {"is_constraint": [True, True, True, False]},
        index=functions.keys(),
        columns=["is_constraint"],
    )

    # create a model instance where some attributes are set to None because they
    # are not needed to create the feasibilty mask
    return InternalModel(
        grids=grids,
        gridspecs=None,
        variable_info=None,
        functions=functions,
        function_info=function_info,
        params=None,
        random_utility_shocks=None,
        n_periods=None,
    )


@pytest.mark.illustrative
def test_get_combined_constraint_illustrative(internal_model_illustrative):
    combined_constraint = get_combined_constraint(internal_model_illustrative)

    age, retirement, lagged_retirement = jnp.array(
        [
            # feasible cases
            [60, 0, 0],  # Young, never retired
            [64, 1, 0],  # Near retirement, newly retired
            [70, 1, 1],  # Properly retired with lagged retirement
            # infeasible cases
            [65, 0, 0],  # Must be retired at 65
            [66, 0, 1],  # Must have lagged retirement at 66
            [60, 0, 1],  # Can't be not retired if was retired before
        ]
    ).T

    # combined constraint expects period not age
    period = age - 18

    exp = jnp.array(3 * [True] + 3 * [False])
    got = combined_constraint(
        period=period, retirement=retirement, lagged_retirement=lagged_retirement
    )
    assert_array_equal(got, exp)


def test_get_multiply_weights():
    multiply_weights = get_multiply_weights(
        stochastic_variables=["a", "b"],
    )

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])

    got = multiply_weights(weight_next_a=a, weight_next_b=b)
    expected = jnp.array([[3, 4], [6, 8]])
    assert_array_equal(got, expected)


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
