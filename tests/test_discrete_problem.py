import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm.discrete_problem import (
    _calculate_emax_extreme_value_shocks,
    _determine_discrete_action_axes_simulation,
    _determine_discrete_action_axes_solution,
    _solve_discrete_problem_no_shocks,
    get_solve_discrete_problem_value,
)
from lcm.typing import ShockType

# ======================================================================================
# Illustrative
# ======================================================================================


@pytest.mark.illustrative
def test_get_solve_discrete_problem_illustrative():
    variable_info = pd.DataFrame(
        {
            "is_action": [False, True],
            "is_state": [True, False],
            "is_discrete": [True, True],
            "is_continuous": [False, False],
        },
    )  # leads to action_axes = [1]

    solve_discrete_problem = get_solve_discrete_problem_value(
        random_utility_shock_type=ShockType.NONE,
        variable_info=variable_info,
        is_last_period=False,
    )

    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )

    got = solve_discrete_problem(cc_values, params={})
    aaae(got, jnp.array([1, 3, 5]))


@pytest.mark.illustrative
def test_solve_discrete_problem_no_shocks_illustrative_single_action_axis():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )
    got = _solve_discrete_problem_no_shocks(
        cc_values,
        action_axes=(0,),
        params={},
    )
    aaae(got, jnp.array([4, 5]))


@pytest.mark.illustrative
def test_solve_discrete_problem_no_shocks_illustrative_multiple_action_axes():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )
    got = _solve_discrete_problem_no_shocks(
        cc_values,
        action_axes=(0, 1),
        params={},
    )
    aaae(got, 5)


@pytest.mark.illustrative
def test_calculate_emax_extreme_value_shocks_illustrative_single_action_axis():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )

    got = _calculate_emax_extreme_value_shocks(
        cc_values,
        action_axes=(0,),
        params={"additive_utility_shock": {"scale": 0.1}},
    )
    aaae(got, jnp.array([4, 5]), decimal=5)


@pytest.mark.illustrative
def test_calculate_emax_extreme_value_shocks_illustrative_multiple_action_axes():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )
    got = _calculate_emax_extreme_value_shocks(
        cc_values,
        action_axes=(0, 1),
        params={"additive_utility_shock": {"scale": 0.1}},
    )
    aaae(got, 5, decimal=5)


# ======================================================================================
# Determine discrete action axes
# ======================================================================================


@pytest.mark.illustrative
def test_determine_discrete_action_axes_illustrative_one_var():
    variable_info = pd.DataFrame(
        {
            "is_action": [False, True],
            "is_state": [True, False],
            "is_discrete": [True, True],
            "is_continuous": [False, False],
        },
    )

    assert _determine_discrete_action_axes_solution(variable_info) == (1,)


@pytest.mark.illustrative
def test_determine_discrete_action_axes_illustrative_three_var():
    variable_info = pd.DataFrame(
        {
            "is_action": [False, True, True, True],
            "is_state": [True, False, False, False],
            "is_discrete": [True, True, True, True],
            "is_continuous": [False, False, False, False],
        },
    )

    assert _determine_discrete_action_axes_solution(variable_info) == (1, 2, 3)


def test_determine_discrete_action_axes():
    variable_info = pd.DataFrame(
        {
            "is_state": [True, True, False, True, False, False],
            "is_action": [False, False, True, True, True, True],
            "is_discrete": [True, True, True, True, True, False],
            "is_continuous": [False, True, False, False, False, True],
        },
    )
    got = _determine_discrete_action_axes_simulation(variable_info)
    assert got == (1, 2, 3)
