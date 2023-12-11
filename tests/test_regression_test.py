import json

import jax.numpy as jnp
import pandas as pd
from lcm._config import TEST_DATA_PATH
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal


def test_regression_test():
    """Test that the output of lcm does not change."""
    # Load generated output
    # ==================================================================================
    with TEST_DATA_PATH.joinpath("regression_tests", "simulation.csv").open() as file:
        expected_simulate = pd.read_csv(file, index_col=["period", "initial_state_id"])

    with TEST_DATA_PATH.joinpath("regression_tests", "solution.json").open() as file:
        _expected_solve = json.load(file)

    # Stack value function array along time dimension
    expected_solve = jnp.stack([jnp.array(data) for data in _expected_solve])

    # Create current lcm ouput
    # ==================================================================================
    model_config = get_model("phelps_deaton_regression_test")

    solve, _ = get_lcm_function(model=model_config.model, targets="solve")

    _got_solve = solve(model_config.params)
    # Stack value function array along time dimension
    got_solve = jnp.stack(_got_solve)

    solve_and_simulate, _ = get_lcm_function(
        model=model_config.model,
        targets="solve_and_simulate",
    )

    got_simulate = solve_and_simulate(
        params=model_config.params,
        initial_states={
            "wealth": jnp.array([1.0, 20, 40, 70]),
        },
    )

    # Compare
    # ==================================================================================
    aaae(expected_solve, got_solve, decimal=5)
    assert_frame_equal(expected_simulate, got_simulate)
