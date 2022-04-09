import jax.numpy as jnp
import numpy as np
import pytest
from lcm.create_state_space import _combine_masks
from lcm.create_state_space import create_combination_grid
from lcm.create_state_space import create_filter_mask
from lcm.create_state_space import create_forward_mask
from lcm.create_state_space import create_indexers_and_segments
from lcm.create_state_space import create_state_choice_space
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS
from numpy.testing import assert_array_almost_equal as aaae


def test_create_state_choice_space_only_simple_variables():
    calculated = create_state_choice_space(PHELPS_DEATON_WITH_SHOCKS)
    expected_value_grid = {
        "retirement": jnp.array([0, 1]),
        "wealth": jnp.linspace(0, 1e4, 11),
    }

    assert calculated["combination_grid"] == {}
    assert set(calculated["value_grid"]) == set(expected_value_grid)
    for key, grid in calculated["value_grid"].items():
        aaae(grid, expected_value_grid[key])


@pytest.fixture()
def filter_mask_inputs():
    def age(period):
        return period + 18

    def mandatory_retirement_filter(retirement, age):
        return jnp.logical_or(retirement == 1, age < 65)

    def mandatory_lagged_retirement_filter(lagged_retirement, age):
        return jnp.logical_or(lagged_retirement == 1, age < 66)

    def absorbing_retirement_filter(retirement, lagged_retirement):
        return jnp.logical_or(retirement == 1, lagged_retirement == 0)

    filters = {
        "mandatory_retirement": mandatory_retirement_filter,
        "mandatory_lagged_retirement": mandatory_lagged_retirement_filter,
        "absorbing_retirement": absorbing_retirement_filter,
    }

    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    out = {"filters": filters, "grids": grids, "aux_functions": {"age": age}}

    return out


PARAMETRIZATION = [
    (50, jnp.array([[False, False], [False, True]])),
    (10, jnp.array([[True, True], [False, True]])),
]


@pytest.mark.parametrize("period, expected", PARAMETRIZATION)
def test_create_filter_mask(filter_mask_inputs, period, expected):

    calculated = create_filter_mask(
        fixed_inputs={"period": period}, **filter_mask_inputs
    )

    aaae(calculated, expected)


def test_create_combination_grid():
    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    mask = jnp.array([[True, False], [True, True]])

    calculated = create_combination_grid(grids=grids, masks=mask)

    expected = {
        "lagged_retirement": jnp.array([0, 1, 1]),
        "retirement": jnp.array([0, 0, 1]),
    }

    for key in expected:
        aaae(calculated[key], expected[key])


def test_create_combination_grid_2_masks():
    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    masks = [
        jnp.array([[True, False], [True, True]]),
        jnp.array([[True, True], [False, True]]),
    ]

    calculated = create_combination_grid(grids=grids, masks=masks)

    expected = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    for key in expected:
        aaae(calculated[key], expected[key])


@pytest.mark.xfail
def test_create_combination_grid_multiple_masks():
    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    masks = [
        jnp.array([[True, False], [True, True]]),
        jnp.array([[True, False], [True, True]]),
        jnp.array([[True, True], [False, True]]),
    ]

    calculated = create_combination_grid(grids=grids, masks=masks)

    expected = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    for key in expected:
        aaae(calculated[key], expected[key])


def test_create_forward_mask():
    """We use the following simplified test case (that does not make economic sense).

    - People can stay at home (work=0), work part time (work=1) or full time (work=2)
    - Experience is measured in work units
    - Initial experience is only [0, 1] but in total one can accumulate up to 6 points
    - People can only work full time if they have no previous work experience
    - People have to work at least part time if they have no previous experience

    """
    grids = {
        "experience": jnp.arange(6),
        "working": jnp.array([0, 1, 2]),
    }

    initial = {
        "experience": jnp.array([0, 0, 1, 1]),
        "working": jnp.array([1, 2, 0, 1]),
    }

    def next_experience(experience, working):
        return experience + working

    calculated = create_forward_mask(
        initial=initial,
        grids=grids,
        next_functions={"next_experience": next_experience},
        jit_next=True,
    )

    expected = jnp.array([False, True, True, False, False, False])

    aaae(calculated, expected)


def test_create_forward_mask_multiple_next_funcs():
    """We use another simple example.

    - People can stay at home (work=0), work part time (work=1) or full time (work=2)
    - Experience is measured in work units
    - Initial experience is only [0, 1] but in total one can accumulate up to 6 points
    - People can only work full time if they have no previous work experience
    - People have to work at least part time if they have no previous experience

    """
    grids = {
        "experience": jnp.arange(6),
        "working": jnp.array([0, 1, 2]),
        "health": jnp.array([0, 1]),
    }

    initial = {
        "experience": jnp.array([0, 0, 1, 1]),
        "working": jnp.array([1, 2, 0, 1]),
        "health": jnp.array([0, 1, 0, 1]),
    }

    def next_experience(experience, working):
        return experience + working

    def next_health(experience, working):
        return int(experience + working > 4)

    calculated = create_forward_mask(
        initial=initial,
        grids=grids,
        next_functions={"next_experience": next_experience, "next_health": next_health},
        jit_next=True,
    )

    expected = jnp.array([False, True, True, False, False, False])

    aaae(calculated, expected)


def test_create_indexers_and_segments():
    mask = np.full((3, 3, 2), False)
    mask[1, 0, 0] = True
    mask[1, -1, -1] = True
    mask[2] = True
    mask = jnp.array(mask)

    state_indexer, choice_indexer, segments = create_indexers_and_segments(
        mask=mask, n_states=2
    )

    expected_state_indexer = jnp.array([[-1, -1, -1], [0, -1, 1], [2, 3, 4]])

    expected_choice_indexer = jnp.array([[0, -1], [-1, 1], [2, 3], [4, 5], [6, 7]])

    expected_segments = jnp.array([0, 1, 2, 2, 3, 3, 4, 4])

    aaae(state_indexer, expected_state_indexer)
    aaae(choice_indexer, expected_choice_indexer)
    aaae(segments, expected_segments)


def test_combine_masks_single_mask():
    mask = jnp.array([True, False])
    calculated = _combine_masks(mask)
    aaae(calculated, mask)


def test_combine_mask_same_shape():
    masks = [
        jnp.array([[True, False], [True, True]]),
        jnp.array([[True, True], [True, False]]),
        jnp.array([[True, True], [True, True]]),
    ]

    expected = jnp.array([[True, False], [True, False]])

    calculated = _combine_masks(masks)
    aaae(calculated, expected)


def test_combine_masks_different_shape():
    masks = [
        jnp.array([[True, False], [True, True], [False, True]]),
        jnp.array([True, True, False]),
    ]

    expected = jnp.array([[True, False], [True, True], [False, False]])

    calculated = _combine_masks(masks)

    aaae(calculated, expected)


def test_combine_masks_invalid_shapes():
    masks = [
        jnp.array([[True, False], [True, True], [False, True]]),
        jnp.array([True, True]),
    ]

    with pytest.raises(TypeError):
        _combine_masks(masks, match="incompatible shapes for broadcasting")
