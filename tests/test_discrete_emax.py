from itertools import product

import jax.numpy as jnp
import pytest
from lcm.discrete_emax import get_emax_calculator
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture()
def values():
    v_t = jnp.arange(20).reshape(2, 2, 5) / 2
    # reuse old test case from when segment axis was last
    v = jnp.transpose(v_t, axes=(2, 0, 1))
    return v


@pytest.fixture()
def segment_info():
    info = {
        "segment_ids": jnp.array([0, 0, 1, 1, 1]),
        "num_segments": 2,
    }
    return info


test_cases = list(product([True, False], range(3)))


@pytest.mark.parametrize("collapse, n_extra_axes", test_cases)
def test_aggregation_without_shocks(values, segment_info, collapse, n_extra_axes):
    values, agg_axes = _get_reshaped_values_and_agg_axes(values, collapse, n_extra_axes)

    calculator = get_emax_calculator(
        shock_type=None,
        choice_axes=agg_axes,
    )

    calculated = calculator(
        values=values,
        choice_segments=segment_info,
        params={},
    )

    expected = jnp.array([8, 9.5])

    expected_shape = tuple([2] + [1] * n_extra_axes)
    assert calculated.shape == expected_shape
    aaae(calculated.flatten(), expected)


scaling_factors = [0.3, 0.6, 1, 2.5, 10]
expected_results = [
    [8.051974, 9.560844],
    [8.225906, 9.800117],
    [8.559682, 10.265875],
    [10.595821, 12.880228],
    [25.184761, 30.494621],
]
test_cases = []
for scale, exp in zip(scaling_factors, expected_results):
    for collapse in [True, False]:
        for n_axes in range(3):
            test_cases.append((scale, exp, collapse, n_axes))


@pytest.mark.parametrize("scale, expected, collapse, n_extra_axes", test_cases)
def test_aggregation_with_extreme_value_shocks(
    values, segment_info, scale, expected, collapse, n_extra_axes
):
    values, agg_axes = _get_reshaped_values_and_agg_axes(values, collapse, n_extra_axes)

    calculator = get_emax_calculator(
        shock_type="extreme_value",
        choice_axes=agg_axes,
    )

    calculated = calculator(
        values=values,
        choice_segments=segment_info,
        params={"additive_utility_shock": {"scale": scale}},
    )

    expected_shape = tuple([2] + [1] * n_extra_axes)
    assert calculated.shape == expected_shape
    aaae(calculated.flatten(), jnp.array(expected), decimal=5)


def _get_reshaped_values_and_agg_axes(values, collapse, n_extra_axes):
    if collapse:
        values = values.reshape(5, 4)
        agg_axes = -1
    else:
        agg_axes = (-2, -1)

    new_shape = tuple([values.shape[0]] + [1] * n_extra_axes + list(values.shape[1:]))
    values = values.reshape(new_shape)

    return values, agg_axes
