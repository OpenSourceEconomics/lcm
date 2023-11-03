"""Test analytical solution and simulation with only discrete choices."""

from copy import deepcopy

import jax.numpy as jnp
import lcm
import numpy as np
import pandas as pd
import pytest
from lcm.entry_point import get_lcm_function
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal


# ======================================================================================
# Model specification
# ======================================================================================
def utility(consumption, working, wealth, productivity):  # noqa: ARG001
    return jnp.log(consumption + 1) - 0.5 * working


def next_wealth(wealth, consumption, working):
    return wealth - consumption + working


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


DETERMINISTIC_MODEL = {
    "functions": {
        "utility": utility,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
    },
    "choices": {
        "consumption": {"options": [0, 1]},
        "working": {"options": [0, 1]},
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 0,
            "stop": 2,
            "n_points": None,
        },
    },
    "n_periods": 2,
}


@lcm.mark.stochastic
def next_productivity(productivity):  # noqa: ARG001
    pass


# ======================================================================================
# Analytical solution and simulation (deterministic model)
# ======================================================================================
def value_second_period(wealth):
    """Value function in the second (last) period. Computed using pen and paper."""
    consumption = np.minimum(1, np.floor(wealth))
    return np.log(consumption + 1)


def policy_second_period(wealth):
    """Policy function in the second (last) period. Computed using pen and paper.

    First column corresponds to consumption choice, second to working choice.

    """
    policy = np.column_stack(
        (np.minimum(1, np.floor(wealth)), np.zeros_like(wealth)),
    ).astype(int)
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "working"])


def value_first_period(wealth, params):
    """Value function in the first period. Computed using pen and paper."""
    index = np.floor(wealth).astype(int)  # map wealth to index 0, 1 and 2
    values = np.array(
        [
            np.maximum(0, params["beta"] * np.log(2) - 0.5),
            np.maximum(0, params["beta"] * np.log(2) - 0.5) + np.log(2),
            (1 + params["beta"]) * np.log(2),
        ],
    )
    return values[index]


def policy_first_period(wealth, params):
    """Policy function in the first period. Computed using pen and paper."""
    index = np.floor(wealth).astype(int)  # map wealth to index 0, 1 and 2
    policies = np.array(
        [
            [0, np.argmax((0, params["beta"] * np.log(2) - 0.5))],
            [1, np.argmax((0, params["beta"] * np.log(2) - 0.5))],
            [1, 0],
        ],
        dtype=int,
    )
    policy = policies[index]
    return matrix_to_dict_of_vectors(policy, col_names=["consumption", "working"])


def matrix_to_dict_of_vectors(arr, col_names):
    """Transform a matrix into a dict of vectors."""
    if arr.ndim != 2:
        raise ValueError("arr must be a two-dimensional array (matrix).")
    return dict(zip(col_names, arr.transpose(), strict=True))


def analytical_solve(wealth_grid, params):
    vf_arr_0 = value_first_period(wealth_grid, params)
    vf_arr_1 = value_second_period(wealth_grid)
    return [vf_arr_0, vf_arr_1]


def analytical_simulate(initial_wealth, params):
    # Simulate
    # ==================================================================================
    vf_arr_0 = value_first_period(initial_wealth, params=params)
    policy_0 = policy_first_period(initial_wealth, params=params)

    wealth_1 = next_wealth(initial_wealth, **policy_0)

    vf_arr_1 = value_second_period(wealth_1)
    policy_1 = policy_second_period(wealth_1)

    policy_0_renamed = {f"{k}_0": v for k, v in policy_0.items()}
    policy_1_renamed = {f"{k}_1": v for k, v in policy_1.items()}

    # Transform data into format as expected by LCM
    # ==================================================================================
    data = (
        {
            "initial_state_id": jnp.arange(len(initial_wealth)),
            "wealth_0": initial_wealth,
            "wealth_1": wealth_1,
            "value_0": vf_arr_0,
            "value_1": vf_arr_1,
        }
        | policy_0_renamed
        | policy_1_renamed
    )

    raw = pd.DataFrame(data)
    raw_long = pd.wide_to_long(
        raw,
        stubnames=["value", "wealth", "consumption", "working"],
        i="initial_state_id",
        j="period",
        sep="_",
    )
    raw_long_with_index = raw_long.swaplevel().sort_index()
    return raw_long_with_index.assign(
        _period=raw_long_with_index.index.get_level_values("period"),
    )


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("beta", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
def test_deterministic_solve(beta, n_wealth_points):
    # Update model
    # ==================================================================================
    model = deepcopy(DETERMINISTIC_MODEL)
    model["states"]["wealth"]["n_points"] = n_wealth_points

    # Solve model using LCM
    # ==================================================================================
    solve, _ = get_lcm_function(
        model=model,
        targets="solve",
    )
    params = {"beta": beta, "utility": {"productivity": 1}}
    got = solve(params)

    # Compute analytical solution
    # ==================================================================================
    wealth_grid_spec = model["states"]["wealth"]
    wealth_grid = np.linspace(
        start=wealth_grid_spec["start"],
        stop=wealth_grid_spec["stop"],
        num=wealth_grid_spec["n_points"],
    )
    expected = analytical_solve(wealth_grid, params=params)

    # Do not assert that in the first period, the arrays have the same values on the
    # first and last index: TODO (@timmens): THIS IS A BUG AND NEEDS TO BE INVESTIGATED.
    # ==================================================================================
    aaae(got[0][slice(1, -1)], expected[0][slice(1, -1)], decimal=12)
    aaae(got[1][slice(None)], expected[1][slice(None)], decimal=12)


@pytest.mark.parametrize("beta", [0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_wealth_points", [100, 1_000])
def test_deterministic_simulate(beta, n_wealth_points):
    # Update model
    # ==================================================================================
    model = deepcopy(DETERMINISTIC_MODEL)
    model["states"]["wealth"]["n_points"] = n_wealth_points

    # Simulate model using LCM
    # ==================================================================================
    solve_and_simulate, _ = get_lcm_function(
        model=model,
        targets="solve_and_simulate",
    )
    params = {"beta": beta, "utility": {"productivity": 1}}
    got = solve_and_simulate(
        params=params,
        initial_states={"wealth": jnp.array([0.25, 0.75, 1.25, 1.75])},
    )

    # Compute analytical simulation
    # ==================================================================================
    expected = analytical_simulate(
        initial_wealth=np.array([0.25, 0.75, 1.25, 1.75]),
        params=params,
    )
    assert_frame_equal(got, expected, check_like=True)


@pytest.mark.skip()
def test_stochastic_solve():
    solve, _ = get_lcm_function(
        model=DETERMINISTIC_MODEL,
        targets="solve",
    )
    transition = jnp.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
        ],
    )
    solve(
        params={"beta": 0.9, "shocks": {"productivity": transition}},
    )
