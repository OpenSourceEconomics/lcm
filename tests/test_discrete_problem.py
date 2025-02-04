import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm.discrete_problem import (
    _calculate_emax_extreme_value_shocks,
    _determine_dense_discrete_choice_axes,
    _solve_discrete_problem_no_shocks,
    get_solve_discrete_problem,
)
from lcm.typing import ShockType

# ======================================================================================
# Illustrative
# ======================================================================================


@pytest.mark.illustrative
def test_get_solve_discrete_problem_illustrative():
    variable_info = pd.DataFrame(
        {
            "is_choice": [False, True],
            "is_continuous": [False, False],
        },
    )  # leads to choice_axes = [1]

    solve_discrete_problem = get_solve_discrete_problem(
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

    got = solve_discrete_problem(cc_values, params=None)
    aaae(got, jnp.array([1, 3, 5]))


@pytest.mark.illustrative
def test_solve_discrete_problem_no_shocks_illustrative():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )

    # Single choice axes
    # ==================================================================================
    got = _solve_discrete_problem_no_shocks(
        cc_values,
        choice_axes=0,
        params=None,
    )
    aaae(got, jnp.array([4, 5]))

    # Tuple of choice axes
    # ==================================================================================
    got = _solve_discrete_problem_no_shocks(
        cc_values,
        choice_axes=(0, 1),
        params=None,
    )
    aaae(got, 5)


@pytest.mark.illustrative
def test_calculate_emax_extreme_value_shocks_illustrative():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )

    # Single choice axes
    # ==================================================================================
    got = _calculate_emax_extreme_value_shocks(
        cc_values,
        choice_axes=0,
        params={"additive_utility_shock": {"scale": 0.1}},
    )
    aaae(got, jnp.array([4, 5]), decimal=5)

    # Tuple of choice axes
    # ==================================================================================
    got = _calculate_emax_extreme_value_shocks(
        cc_values,
        choice_axes=(0, 1),
        params={"additive_utility_shock": {"scale": 0.1}},
    )
    aaae(got, 5, decimal=5)


# ======================================================================================
# Determine discrete choice axes
# ======================================================================================


@pytest.mark.illustrative
def test_determine_discrete_choice_axes_illustrative_one_var():
    variable_info = pd.DataFrame(
        {
            "is_choice": [False, True],
            "is_continuous": [False, False],
        },
    )

    assert _determine_dense_discrete_choice_axes(variable_info) == (1,)


@pytest.mark.illustrative
def test_determine_discrete_choice_axes_illustrative_three_var():
    variable_info = pd.DataFrame(
        {
            "is_choice": [False, True, True, True],
            "is_continuous": [False, False, False, False],
        },
    )

    assert _determine_dense_discrete_choice_axes(variable_info) == (1, 2, 3)
