from functools import partial
from itertools import product

import jax.numpy as jnp
import pandas as pd
import pytest
from jax.ops import segment_max
from numpy.testing import assert_array_almost_equal as aaae

from lcm.discrete_problem import (
    _calculate_emax_extreme_value_shocks,
    _determine_dense_discrete_choice_axes,
    _segment_extreme_value_emax_over_first_axis,
    _segment_logsumexp,
    _solve_discrete_problem_no_shocks,
    get_solve_discrete_problem,
)
from lcm.typing import ShockType


@pytest.fixture
def cc_values():
    """Conditional continuation values."""
    v_t = jnp.arange(20).reshape(2, 2, 5) / 2
    # reuse old test case from when segment axis was last
    return jnp.transpose(v_t, axes=(2, 0, 1))


@pytest.fixture
def segment_info():
    return {
        "segment_ids": jnp.array([0, 0, 1, 1, 1]),
        "num_segments": 2,
    }


test_cases = list(product([True, False], range(3)))

# ======================================================================================
# Aggregation without shocks
# ======================================================================================


@pytest.mark.xfail(reason="Removec choice segments")
@pytest.mark.parametrize(("collapse", "n_extra_axes"), test_cases)
def test_aggregation_without_shocks(cc_values, segment_info, collapse, n_extra_axes):
    cc_values, var_info = _get_reshaped_cc_values_and_variable_info(
        cc_values,
        collapse,
        n_extra_axes,
    )

    solve_discrete_problem = get_solve_discrete_problem(
        random_utility_shock_type=ShockType.NONE,
        variable_info=var_info,
        is_last_period=False,
        choice_segments=segment_info,
    )

    calculated = solve_discrete_problem(cc_values, params=None)

    expected = jnp.array([8, 9.5])

    expected_shape = tuple([2] + [1] * n_extra_axes)
    assert calculated.shape == expected_shape
    aaae(calculated.flatten(), expected)


# ======================================================================================
# Aggregation with extreme value shocks
# ======================================================================================

scaling_factors = [0.3, 0.6, 1, 2.5, 10]
expected_results = [
    [8.051974, 9.560844],
    [8.225906, 9.800117],
    [8.559682, 10.265875],
    [10.595821, 12.880228],
    [25.184761, 30.494621],
]
test_cases = []
for scale, exp in zip(scaling_factors, expected_results, strict=True):
    for collapse in [True, False]:
        for n_axes in range(3):
            test_cases.append((scale, exp, collapse, n_axes))


@pytest.mark.parametrize(("scale", "expected", "collapse", "n_extra_axes"), test_cases)
def test_aggregation_with_extreme_value_shocks(
    cc_values,
    segment_info,
    scale,
    expected,
    collapse,
    n_extra_axes,
):
    cc_values, var_info = _get_reshaped_cc_values_and_variable_info(
        cc_values,
        collapse,
        n_extra_axes,
    )

    choice_axes = _determine_dense_discrete_choice_axes(var_info)
    solve_discrete_problem = partial(
        _calculate_emax_extreme_value_shocks,
        choice_axes=choice_axes,
        choice_segments=segment_info,
        params={"additive_utility_shock": {"scale": scale}},
    )

    calculated = solve_discrete_problem(cc_values)

    expected_shape = tuple([2] + [1] * n_extra_axes)
    assert calculated.shape == expected_shape
    aaae(calculated.flatten(), jnp.array(expected), decimal=5)


def _get_reshaped_cc_values_and_variable_info(cc_values, collapse, n_extra_axes):
    n_variables = cc_values.ndim + 1 + n_extra_axes - collapse
    n_agg_axes = 1 if collapse else 2
    names = [f"v{i}" for i in range(n_variables)]
    is_choice = [False, True] + [False] * n_extra_axes + [True] * n_agg_axes
    is_sparse = [True, True] + [False] * (n_variables - 2)
    var_info = pd.DataFrame(index=names)
    var_info["is_choice"] = is_choice
    var_info["is_sparse"] = is_sparse
    var_info["is_dense"] = ~var_info["is_sparse"]
    var_info["is_continuous"] = False

    if collapse:
        cc_values = cc_values.reshape(5, 4)

    new_shape = tuple(
        [cc_values.shape[0]] + [1] * n_extra_axes + list(cc_values.shape[1:]),
    )
    cc_values = cc_values.reshape(new_shape)

    return cc_values, var_info


# ======================================================================================
# Illustrative
# ======================================================================================


@pytest.mark.illustrative
def test_get_solve_discrete_problem_illustrative():
    variable_info = pd.DataFrame(
        {
            "is_sparse": [True, False, False],
            "is_dense": [False, True, True],
            "is_choice": [True, True, False],
            "is_continuous": [False, False, False],
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


@pytest.mark.xfail(reason="Removec choice segments")
@pytest.mark.illustrative
def test_solve_discrete_problem_no_shocks_illustrative():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )

    # Only choice axes
    # ==================================================================================
    got = _solve_discrete_problem_no_shocks(
        cc_values,
        choice_axes=0,
        params=None,
    )
    aaae(got, jnp.array([4, 5]))

    # Only choice segment
    # ==================================================================================
    got = _solve_discrete_problem_no_shocks(
        cc_values,
        choice_axes=None,
        params=None,
    )
    aaae(got, jnp.array([[2, 3], [4, 5]]))

    # Choice axes and choice segment
    # ==================================================================================
    got = _solve_discrete_problem_no_shocks(
        cc_values,
        choice_axes=1,
        params=None,
    )
    aaae(got, jnp.array([3, 5]))


@pytest.mark.illustrative
def test_calculate_emax_extreme_value_shocks_illustrative():
    cc_values = jnp.array(
        [
            [0, 1],
            [2, 3],
            [4, 5],
        ],
    )

    # Only choice axes
    # ==================================================================================
    got = _calculate_emax_extreme_value_shocks(
        cc_values,
        choice_axes=0,
        choice_segments=None,
        params={"additive_utility_shock": {"scale": 0.1}},
    )
    aaae(got, jnp.array([4, 5]), decimal=5)

    # Only choice segment
    # ==================================================================================
    got = _calculate_emax_extreme_value_shocks(
        cc_values,
        choice_axes=None,
        choice_segments={"segment_ids": jnp.array([0, 0, 1]), "num_segments": 2},
        params={"additive_utility_shock": {"scale": 0.1}},
    )
    aaae(got, jnp.array([[2, 3], [4, 5]]), decimal=5)

    # Choice axes and choice segment
    # ==================================================================================
    got = _calculate_emax_extreme_value_shocks(
        cc_values,
        choice_axes=1,
        choice_segments={"segment_ids": jnp.array([0, 0, 1]), "num_segments": 2},
        params={"additive_utility_shock": {"scale": 0.1}},
    )
    aaae(got, jnp.array([3, 5]), decimal=5)


# ======================================================================================
# Segment max over first axis
# ======================================================================================


@pytest.mark.illustrative
def test_segment_max_over_first_axis_illustrative():
    a = jnp.arange(4)
    segment_info = {
        "segment_ids": jnp.array([0, 0, 1, 1]),
        "num_segments": 2,
    }
    got = segment_max(a, indices_are_sorted=True, **segment_info)
    expected = jnp.array([1, 3])
    aaae(got, expected)


@pytest.mark.illustrative
def test_segment_extreme_value_emax_over_first_axis_illustrative():
    a = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

    segment_info = {
        "segment_ids": jnp.array([0, 0, 0, 1, 1, 1]),
        "num_segments": 2,
    }

    got = _segment_extreme_value_emax_over_first_axis(
        a,
        scale=0.1,
        segment_info=segment_info,
    )
    expected = jnp.array([[4, 5], [10, 11]])
    aaae(got, expected)


@pytest.mark.illustrative
def test_segment_logsumexp_illustrative():
    a = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])

    segment_info = {
        "segment_ids": jnp.array([0, 0, 0, 1, 1, 1]),
        "num_segments": 2,
    }

    got = _segment_logsumexp(a, segment_info)
    expected = jnp.array([[4, 5], [10, 11]])
    aaae(got, expected, decimal=0)


# ======================================================================================
# Determine discrete choice axes
# ======================================================================================


@pytest.mark.illustrative
def test_determine_discrete_choice_axes_illustrative():
    # No discrete choice variable
    # ==================================================================================

    variable_info = pd.DataFrame(
        {
            "is_sparse": [True, False],
            "is_dense": [False, True],
            "is_choice": [True, False],
            "is_discrete": [True, True],
            "is_continuous": [False, False],
        },
    )

    assert _determine_dense_discrete_choice_axes(variable_info) is None

    # One discrete choice variable
    # ==================================================================================

    variable_info = pd.DataFrame(
        {
            "is_sparse": [True, False, False, False],
            "is_dense": [False, True, True, True],
            "is_choice": [True, True, False, True],
            "is_discrete": [True, True, True, True],
            "is_continuous": [False, False, False, False],
        },
    )

    assert _determine_dense_discrete_choice_axes(variable_info) == (1, 3)
