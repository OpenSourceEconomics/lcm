import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from lcm.example_models import PHELPS_DEATON_WITH_FILTERS
from lcm.interfaces import Model
from lcm.process_model import process_model
from lcm.state_space import (
    create_combination_grid,
    create_filter_mask,
    create_forward_mask,
    create_indexers_and_segments,
    create_state_choice_space,
)
from numpy.testing import assert_array_almost_equal as aaae


def test_create_state_choice_space():
    _model = process_model(PHELPS_DEATON_WITH_FILTERS)
    create_state_choice_space(model=_model, period=0, jit_filter=False)


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

    grids = {
        "lagged_retirement": jnp.array([0, 1]),
        "retirement": jnp.array([0, 1]),
    }

    functions = {
        "mandatory_retirement_filter": mandatory_retirement_filter,
        "mandatory_lagged_retirement_filter": mandatory_lagged_retirement_filter,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "age": age,
    }

    function_info = pd.DataFrame(
        index=functions.keys(),
        columns=["is_filter"],
        data=[[True], [True], [True], [False]],
    )

    # create a model instance where some attributes are set to None because they
    # are not needed for create_filter_mask
    model = Model(
        grids=grids,
        gridspecs=None,
        variable_info=None,
        functions=functions,
        function_info=function_info,
        shocks=None,
        n_periods=100,
        params={},
    )

    return model


PARAMETRIZATION = [
    (50, jnp.array([[False, False], [False, True]])),
    (10, jnp.array([[True, True], [False, True]])),
]


@pytest.mark.parametrize(("period", "expected"), PARAMETRIZATION)
def test_create_filter_mask(filter_mask_inputs, period, expected):
    calculated = create_filter_mask(
        model=filter_mask_inputs,
        subset=["lagged_retirement", "retirement"],
        fixed_inputs={"period": period},
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
    - People get bad health after they have more than 1 experience

    """
    grids = {
        "experience": jnp.arange(6),
        "working": jnp.array([0, 1, 2]),
        "health": jnp.array([0, 1]),
    }

    initial = {
        "experience": jnp.array([0, 0, 1, 1]),
        "working": jnp.array([1, 2, 0, 1]),
        "health": jnp.array([0, 0, 0, 0]),
    }

    def next_experience(experience, working):
        return experience + working

    def next_health(experience, working):
        return ((experience + working) > 1).astype(int)

    calculated = create_forward_mask(
        initial=initial,
        grids=grids,
        next_functions={"next_experience": next_experience, "next_health": next_health},
        jit_next=True,
    )

    expected = jnp.array(
        [
            [False, False],
            [True, False],
            [False, True],
            [False, False],
            [False, False],
            [False, False],
        ],
    )

    aaae(calculated, expected)


def test_forward_mask_w_aux_function():
    """We use another simple example.

    - People can stay at home (work=0), work part time (work=1) or full time (work=2)
    - Experience is measured in work units
    - Initial experience is only [0, 1] but in total one can accumulate up to 6 points
    - People have to work at least part time if they have no previous experience
    - People get bad health after they work full time.
    - In bad health additional work experience does not add anything to experience.

    """
    grids = {
        "experience": jnp.arange(6),
        "working": jnp.array([0, 1, 2]),
        "health": jnp.array([0, 1]),
    }

    initial = {
        "experience": jnp.array([0, 0, 1, 1, 2]),
        "working": jnp.array([1, 2, 0, 1, 2]),
        "health": jnp.array([0, 0, 1, 1, 1]),
    }

    def healthy_working(health, working):
        return jnp.where(health == 0, working, 0)

    def next_experience(experience, healthy_working):
        return experience + healthy_working

    def next_health(working):
        return (working == 2).astype(int)

    calculated = create_forward_mask(
        initial=initial,
        grids=grids,
        next_functions={"next_experience": next_experience, "next_health": next_health},
        aux_functions={"healthy_working": healthy_working},
        jit_next=False,
    )

    expected = jnp.array(
        [
            [False, False],
            [True, False],
            [False, True],
            [False, False],
            [False, False],
            [False, False],
        ],
    )

    aaae(calculated, expected)


def test_create_indexers_and_segments():
    mask = np.full((3, 3, 2), fill_value=False)
    mask[1, 0, 0] = True
    mask[1, -1, -1] = True
    mask[2] = True
    mask = jnp.array(mask)

    state_indexer, choice_indexer, segments = create_indexers_and_segments(
        mask=mask,
        n_sparse_states=2,
    )

    expected_state_indexer = jnp.array([[-1, -1, -1], [0, -1, 1], [2, 3, 4]])

    expected_choice_indexer = jnp.array([[0, -1], [-1, 1], [2, 3], [4, 5], [6, 7]])

    expected_segments = jnp.array([0, 1, 2, 2, 3, 3, 4, 4])

    aaae(state_indexer, expected_state_indexer)
    aaae(choice_indexer, expected_choice_indexer)
    aaae(segments["segment_ids"], expected_segments)
