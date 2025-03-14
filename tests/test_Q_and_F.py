import jax.numpy as jnp
import pandas as pd
import pytest
from jax import Array
from numpy.testing import assert_array_equal

from lcm.input_processing import process_model
from lcm.interfaces import InternalModel
from lcm.Q_and_F import (
    _get_feasibility,
    _get_joint_weights_function,
    get_Q_and_F,
)
from lcm.state_action_space import create_state_space_info
from lcm.typing import ShockType
from tests.test_models import get_model_config
from tests.test_models.deterministic import utility


@pytest.mark.illustrative
def test_get_Q_and_F_function():
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

    state_space_info = create_state_space_info(
        model=model,
        is_last_period=False,
    )

    Q_and_F = get_Q_and_F(
        model=model,
        next_state_space_info=state_space_info,
        period=model.n_periods - 1,
    )

    consumption = jnp.array([10, 20, 30])
    retirement = jnp.array([0, 1, 0])
    wealth = jnp.array([20, 20, 20])

    Q, F = Q_and_F(
        consumption=consumption,
        retirement=retirement,
        wealth=wealth,
        params=params,
        vf_arr=None,
    )

    assert_array_equal(
        Q,
        utility(
            consumption=consumption,
            working=1 - retirement,
            disutility_of_work=1.0,
        ),
    )
    assert_array_equal(F, jnp.array([True, True, False]))


@pytest.fixture
def internal_model_illustrative():
    def age(period):
        return period + 18

    def mandatory_retirement_constraint(retirement, age, params):  # noqa: ARG001
        # Individuals must be retired from age 65 onwards
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_constraint(lagged_retirement, age, params):  # noqa: ARG001
        # Individuals must have been retired last year from age 66 onwards
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_constraint(retirement, lagged_retirement, params):  # noqa: ARG001
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
        index=list(functions),
    )

    # create a model instance where some attributes are set to None because they
    # are not needed to create the feasibilty mask
    return InternalModel(
        grids=grids,
        gridspecs={},
        variable_info=pd.DataFrame(),
        functions=functions,  # type: ignore[arg-type]
        function_info=function_info,
        params={},
        random_utility_shocks=ShockType.NONE,
        n_periods=0,
    )


@pytest.mark.illustrative
def test_get_combined_constraint_illustrative(internal_model_illustrative):
    combined_constraint = _get_feasibility(internal_model_illustrative)

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
        period=period,
        retirement=retirement,
        lagged_retirement=lagged_retirement,
        params={},
    )
    assert_array_equal(got, exp)


def test_get_multiply_weights():
    multiply_weights = _get_joint_weights_function(
        stochastic_variables=["a", "b"],
    )

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])

    got = multiply_weights(weight_next_a=a, weight_next_b=b)
    expected = jnp.array([[3, 4], [6, 8]])
    assert_array_equal(got, expected)


def test_get_combined_constraint():
    def f(params):  # noqa: ARG001
        return True

    def g(params):  # noqa: ARG001
        return False

    def h(params):  # noqa: ARG001
        return None

    function_info = pd.DataFrame(
        {"is_constraint": [True, True, False]},
        index=["f", "g", "h"],
    )
    model = InternalModel(
        grids={},
        gridspecs={},
        variable_info=pd.DataFrame(),
        functions={"f": f, "g": g, "h": h},  # type: ignore[dict-item]
        function_info=function_info,
        params={},
        random_utility_shocks=ShockType.NONE,
        n_periods=0,
    )
    combined_constraint = _get_feasibility(model)
    feasibility: Array = combined_constraint(params={})  # type: ignore[assignment]
    assert feasibility.item() is False
