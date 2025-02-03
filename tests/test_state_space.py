import jax.numpy as jnp
import pandas as pd
import pytest

from lcm.input_processing import process_model
from lcm.interfaces import InternalModel
from lcm.state_space import (
    create_state_choice_space,
)
from tests.test_models import get_model_config


def test_create_state_choice_space():
    _model = process_model(
        get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )
    create_state_choice_space(
        model=_model,
        is_last_period=False,
    )


@pytest.fixture
def filter_mask_inputs():
    def age(period):
        return period + 18

    def mandatory_retirement_filter(retirement, age):
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_filter(lagged_retirement, age):
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_filter(retirement, lagged_retirement):
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    functions = {
        "mandatory_retirement_filter": mandatory_retirement_filter,
        "mandatory_lagged_retirement_filter": mandatory_lagged_retirement_filter,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "age": age,
    }

    function_info = pd.DataFrame(
        index=functions.keys(),
        columns=["is_filter"],
        data=[[True], [True], [True], [False]],
    )

    # create a model instance where some attributes are set to None because they
    # are not needed for create_filter_mask
    return InternalModel(
        grids=grids,
        gridspecs=None,
        variable_info=None,
        functions=functions,
        function_info=function_info,
        params=None,
        random_utility_shocks=None,
        n_periods=100,
    )


PARAMETRIZATION = [
    (50, jnp.array([[False, False], [False, True]])),
    (10, jnp.array([[True, True], [False, True]])),
]
