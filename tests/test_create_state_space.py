import numpy as np
from lcm.create_state_space import create_state_choice_space
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS
from numpy.testing import assert_array_almost_equal as aaae


def test_create_state_choice_space_only_simple_variables():
    calculated = create_state_choice_space(PHELPS_DEATON_WITH_SHOCKS)
    expected_value_grid = {
        "retirement": np.array([0, 1]),
        "wealth": np.linspace(0, 1e6, 12),
    }

    assert calculated["combination_grid"] == {}
    assert set(calculated["value_grid"]) == set(expected_value_grid)
    for key, grid in calculated["value_grid"].items():
        aaae(grid, expected_value_grid[key])
