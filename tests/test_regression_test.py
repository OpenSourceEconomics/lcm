import jax.numpy as jnp
import pandas as pd
from lcm._config import TEST_DATA
from lcm.entry_point import get_lcm_function
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from tests.test_models.phelps_deaton import PHELPS_DEATON

REGRESSION_TEST_MODEL = {**PHELPS_DEATON, "n_perids": 5}
REGRESSION_TEST_PARAMS = {
    "beta": 0.95,
    "utility": {"disutility_of_work": 1.0},
    "next_wealth": {
        "interest_rate": 0.05,
    },
}


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
    solve, _ = get_lcm_function(model=REGRESSION_TEST_MODEL, targets="solve")

    got_solve = solve(REGRESSION_TEST_PARAMS)

    solve_and_simulate, _ = get_lcm_function(
        model=REGRESSION_TEST_MODEL,
        targets="solve_and_simulate",
    )

    got_simulate = solve_and_simulate(
        params=REGRESSION_TEST_PARAMS,
        initial_states={
            "wealth": jnp.array([5.0, 20, 40, 70]),
        },
    )

    # ----------------------------------------------------------------------------------
    # Compare
    # ----------------------------------------------------------------------------------
    aaae(expected_solve, got_solve, decimal=5)
    assert_frame_equal(expected_simulate, got_simulate)
