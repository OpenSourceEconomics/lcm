import jax.numpy as jnp
import pytest
from lcm.create_state_space import create_filter_mask
from lcm.create_state_space import create_state_choice_space
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS
from numpy.testing import assert_array_almost_equal as aaae


def test_create_state_choice_space_only_simple_variables():
    calculated = create_state_choice_space(PHELPS_DEATON_WITH_SHOCKS)
    expected_value_grid = {
        "retirement": jnp.array([0, 1]),
        "wealth": jnp.linspace(0, 1e4, 11),
    }

    assert calculated["combination_grid"] == {}
    assert set(calculated["value_grid"]) == set(expected_value_grid)
    for key, grid in calculated["value_grid"].items():
        aaae(grid, expected_value_grid[key])


@pytest.fixture()
def filter_mask_inputs():
    def age(period):
        return period + 18

    def mandatory_retirement_filter(retirement, age):
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_filter(lagged_retirement, age):
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_filter(retirement, lagged_retirement):
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    filters = {
        "mandatory_retirement": mandatory_retirement_filter,
        "mandatory_lagged_retirement": mandatory_lagged_retirement_filter,
        "absorbing_retirement": absorbing_retirement_filter,
    }

    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    out = {"filters": filters, "grids": grids, "aux_functions": {"age": age}}

    return out


PARAMETRIZATION = [
    (50, jnp.array([[False, False], [False, True]])),
    (10, jnp.array([[True, True], [False, True]])),
]


@pytest.mark.parametrize("period, expected", PARAMETRIZATION)
def test_create_filter_mask(filter_mask_inputs, period, expected):

    calculated = create_filter_mask(
        fixed_inputs={"period": period}, **filter_mask_inputs
    )

    aaae(calculated, expected)
