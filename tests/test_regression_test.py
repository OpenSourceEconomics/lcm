import jax.numpy as jnp
import pandas as pd
from lcm._config import TEST_DATA
from lcm.entry_point import get_lcm_function
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from tests.test_models.deterministic import get_model_config, get_params


def test_regression_test():
    """Test that the output of lcm does not change."""
    # ----------------------------------------------------------------------------------
    # Load generated output
    # ----------------------------------------------------------------------------------
    expected_simulate = pd.read_pickle(
        TEST_DATA.joinpath("regression_tests", "simulation.pkl"),
    )

    expected_solve = pd.read_pickle(
        TEST_DATA.joinpath("regression_tests", "solution.pkl"),
    )

    # ----------------------------------------------------------------------------------
    # Generate current lcm ouput
    # ----------------------------------------------------------------------------------
    model_config = get_model_config("stripped_down", n_periods=3)

    solve, _ = get_lcm_function(model=model_config, targets="solve")

    params = get_params(
        beta=0.95,
        disutility_of_work=1.0,
        interest_rate=0.05,
    )
    got_solve = solve(params)

    solve_and_simulate, _ = get_lcm_function(
        model=model_config,
        targets="solve_and_simulate",
    )

    got_simulate = solve_and_simulate(
        params=params,
        initial_states={
            "wealth": jnp.array([5.0, 20, 40, 70]),
        },
    )

    # ----------------------------------------------------------------------------------
    # Compare
    # ----------------------------------------------------------------------------------
    aaae(expected_solve, got_solve, decimal=5)
    assert_frame_equal(expected_simulate, got_simulate)
