import json

import jax.numpy as jnp
from lcm._config import TEST_DATA_PATH
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model
from pybaum import tree_equal, tree_map


def test_regression_test():
    """Test that the output of lcm does not change."""
    # Load generated output
    # ==================================================================================
    with TEST_DATA_PATH.joinpath(
        "regression_tests",
        "solution_and_simulation.json",
    ).open() as file:
        expected = json.load(file)

    # Create current lcm ouput
    # ==================================================================================
    model_config = get_model("phelps_deaton_regression_test")

    solve_model, _ = get_lcm_function(model=model_config.model, targets="solve")
    simulate_model, _ = get_lcm_function(model=model_config.model, targets="simulate")

    solution = solve_model(model_config.params)

    simulation = simulate_model(
        params=model_config.params,
        vf_arr_list=solution,
        initial_states={
            "wealth": jnp.array([1.0, 20, 40, 70]),
        },
    )

    got = {
        "solution": jnp.stack(solution),
        "simulation": simulation,
    }
    # convert all jax.ndarray to list of lists for comparison
    got = tree_map(_convert_to_list_if_array, got)

    # Compare
    # ==================================================================================

    tree_equal(expected, got)


def _convert_to_list_if_array(arr_or_float):
    if isinstance(arr_or_float, jnp.ndarray):
        out = arr_or_float.tolist()
    else:
        out = arr_or_float
    return out
