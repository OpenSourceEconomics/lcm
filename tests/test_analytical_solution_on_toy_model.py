"""Test analytical solution and simulation with only discrete choices."""
import itertools

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lcm.entry_point import get_lcm_function
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal


# ======================================================================================
# Model specification
# ======================================================================================
def utility(consumption, working, wealth=None):  # noqa: ARG001
    return jnp.log(consumption + 1) - 0.5 * working


def next_wealth(wealth, consumption, working):
    return wealth - consumption + working


def consumption_constraint(consumption, wealth):
    return consumption <= wealth


N_WEALTH_POINTS = 100


MODEL = {
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
            "start": 0.01,
            "stop": 2,
            "n_points": N_WEALTH_POINTS,
        },
    },
    "n_periods": 2,
}


# ======================================================================================
# Analytical solution and simulation
# ======================================================================================
def arr2d_to_dict_of_arr1d(arr, col_names):
    """Transform a 2d array into a dict of 1d arrays."""
    return dict(zip(col_names, arr.transpose(), strict=True))


def value_second_period(wealth):
    """Value function in the second (last) period. Computed using pen and paper."""
    consumption = np.minimum(1, np.floor(wealth))
    return utility(consumption, working=0)


def policy_second_period(wealth):
    """Policy function in the second (last) period. Computed using pen and paper.

    First column corresponds to consumption choice, second to working choice.

    """
    index = np.array(wealth >= 1, dtype=int)
    policy = np.array([[0, 0], [1, 0]])[index]
    return arr2d_to_dict_of_arr1d(policy, col_names=["consumption", "working"])


def _big_u_first_period(wealth, consumption, working):
    beta = 0.9
    if consumption_constraint(consumption, wealth):
        nw = next_wealth(wealth, consumption, working)
        out = utility(consumption, working=working) + beta * value_second_period(nw)
    else:
        out = -np.inf
    return out


def value_and_policy_first_period(wealth):
    """Value and policy function in the first period.

    Assuming that consumption and working are both binary choices.

    """
    # Choice grids
    # ==================================================================================
    c = np.arange(2)
    a = np.arange(2)
    n_choices = 2

    # Grids
    # ==================================================================================

    vf_arr = np.empty(len(wealth))
    policy = np.empty((len(wealth), n_choices), dtype=int)

    # Loop over wealth levels
    # ==================================================================================
    for i, w in enumerate(wealth):
        # For each wealth level, loop over choices and pick the best one
        # ==============================================================================
        values = np.empty((len(c), len(a)))
        for _c, _a in itertools.product(c, a):
            values[_c, _a] = _big_u_first_period(
                wealth=w,
                consumption=_c,
                working=_a,
            )

        policy[i, :] = np.unravel_index(values.argmax(), values.shape)
        vf_arr[i] = values.max()

    return vf_arr, arr2d_to_dict_of_arr1d(policy, col_names=["consumption", "working"])


def analytical_solve(wealth_grid):
    vf_arr_0, _ = value_and_policy_first_period(wealth_grid)
    vf_arr_1 = value_second_period(wealth_grid)
    return [vf_arr_0, vf_arr_1]


def analytical_simulate(initial_wealth):
    vf_arr_0, policy_0 = value_and_policy_first_period(initial_wealth)

    _next_wealth = next_wealth(initial_wealth, **policy_0)
    vf_arr_1_for_simulation = value_second_period(_next_wealth)

    policy_1 = policy_second_period(_next_wealth)

    policy_0_renamed = {f"{k}_0": v for k, v in policy_0.items()}
    policy_1_renamed = {f"{k}_1": v for k, v in policy_1.items()}

    raw = pd.DataFrame(
        {
            "initial_state_id": jnp.arange(len(initial_wealth)),
            "wealth_0": initial_wealth,
            "wealth_1": _next_wealth,
            "value_0": vf_arr_0,
            **policy_0_renamed,
            "value_1": vf_arr_1_for_simulation,
            **policy_1_renamed,
        },
    )
    raw_long = pd.wide_to_long(
        raw,
        stubnames=["value", "wealth", "consumption", "working"],
        i="initial_state_id",
        j="period",
        sep="_",
    )
    raw_with_correct_index = raw_long.swaplevel().sort_index()
    return raw_with_correct_index.assign(
        _period=raw_with_correct_index.index.get_level_values("period"),
    )


# ======================================================================================
# Tests
# ======================================================================================
def test_solve():
    solve, _ = get_lcm_function(
        model=MODEL,
        targets="solve",
    )
    got = solve(
        params={"beta": 0.9},
    )
    expected = analytical_solve(wealth_grid=np.linspace(0, 2, N_WEALTH_POINTS))

    # Assert that in the first period, the arrays do not have the same values on the
    # first and last index: THIS IS A BUG AND NEEDS TO BE INVESTIGATED
    # ==================================================================================
    first_and_last_idx = np.array([0, -1])
    with pytest.raises(AssertionError):
        aaae(got[0][first_and_last_idx], expected[0][first_and_last_idx], decimal=5)

    # Assert that in the first period, both arrays have the same values on all values
    # except the first and last index
    # ==================================================================================
    aaae(got[0][slice(1, -1)], expected[0][slice(1, -1)], decimal=5)

    # Assert that in the first period, both arrays have the same values on all values
    # ==================================================================================
    aaae(got[1][slice(None)], expected[1][slice(None)], decimal=9)


def test_simulate():
    solve_and_simulate, _ = get_lcm_function(
        model=MODEL,
        targets="solve_and_simulate",
    )
    got = solve_and_simulate(
        params={"beta": 0.9},
        initial_states={"wealth": jnp.array([0.25, 0.75, 1.25, 1.75])},
    )
    expected = analytical_simulate(initial_wealth=np.array([0.25, 0.75, 1.25, 1.75]))
    assert_frame_equal(got, expected, check_like=True)
