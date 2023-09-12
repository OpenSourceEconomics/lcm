import jax
import jax.numpy as jnp
import lcm.grids as grids_module
import numpy as np
import pandas as pd
import pytest
from lcm.example_models import (
    N_CHOICE_GRID_POINTS,
    N_STATE_GRID_POINTS,
    PHELPS_DEATON,
    PHELPS_DEATON_WITH_FILTERS,
)
from lcm.interfaces import GridSpec
from lcm.process_model import (
    _get_function_info,
    _get_grids,
    _get_gridspecs,
    _get_variable_info,
    process_model,
)
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


@pytest.fixture()
def user_model():
    def f(a, b):
        return a + b

    return {
        "functions": {
            "f": f,
        },
        "choices": {
            "a": {"options": [0, 1]},
        },
        "states": {
            "c": {"options": [2, 3]},
        },
        "n_periods": 2,
    }


def test_get_function_info(user_model):
    got = _get_function_info(user_model)
    exp = pd.DataFrame(
        {
            "is_filter": [False],
            "is_constraint": [False],
            "is_next": [False],
        },
        index=["f"],
    )
    assert_frame_equal(got, exp)


def test_get_variable_info(user_model):
    function_info = _get_function_info(user_model)
    got = _get_variable_info(
        user_model,
        function_info,
    )
    exp = pd.DataFrame(
        {
            "is_state": [False, True],
            "is_choice": [True, False],
            "is_discrete": [True, True],
            "is_continuous": [False, False],
            "is_auxiliary": [False, True],
            "is_sparse": [False, False],
            "is_dense": [True, True],
        },
        index=["a", "c"],
    )
    assert_frame_equal(got.loc[exp.index], exp)  # we don't care about the id order here


def test_get_gridspecs(user_model):
    variable_info = _get_variable_info(
        user_model,
        function_info=_get_function_info(user_model),
    )
    got = _get_gridspecs(user_model, variable_info)
    exp = {"a": [0, 1], "c": [2, 3]}
    assert got == exp


def test_get_grids(user_model):
    variable_info = _get_variable_info(
        user_model,
        function_info=_get_function_info(user_model),
    )
    gridspecs = _get_gridspecs(user_model, variable_info)
    got = _get_grids(gridspecs, variable_info)
    assert_array_equal(got["a"], jnp.array([0, 1]))
    assert_array_equal(got["c"], jnp.array([2, 3]))


def test_process_phelps_deaton_with_filters():
    model = process_model(PHELPS_DEATON_WITH_FILTERS)

    # Variable Info
    assert (
        model.variable_info["is_sparse"].to_numpy()
        == np.array([True, True, False, False])
    ).all()

    assert (
        model.variable_info["is_state"].to_numpy()
        == np.array([True, False, True, False])
    ).all()

    assert (
        model.variable_info["is_continuous"].to_numpy()
        == np.array([False, False, True, True])
    ).all()

    # Gridspecs
    wealth_specs = GridSpec(
        kind="linspace",
        specs={"start": 0, "stop": 100, "n_points": N_STATE_GRID_POINTS},
    )

    assert model.gridspecs["wealth"] == wealth_specs

    consumption_specs = GridSpec(
        kind="linspace",
        specs={"start": 1, "stop": 100, "n_points": N_CHOICE_GRID_POINTS},
    )
    assert model.gridspecs["consumption"] == consumption_specs

    assert model.gridspecs["retirement"] == [0, 1]
    assert model.gridspecs["lagged_retirement"] == [0, 1]

    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].specs)
    assert (asserted == model.grids["consumption"]).all()

    func = getattr(grids_module, model.gridspecs["wealth"].kind)
    asserted = func(**model.gridspecs["wealth"].specs)
    assert (asserted == model.grids["wealth"]).all()

    assert (model.grids["retirement"] == jax.numpy.array([0, 1])).all()
    assert (model.grids["lagged_retirement"] == jax.numpy.array([0, 1])).all()

    # Functions
    assert (
        model.function_info["is_next"].to_numpy()
        == np.array([False, True, False, False, False, True])
    ).all()
    assert ~model.function_info.loc["utility"].to_numpy().any()


def test_process_phelps_deaton():
    model = process_model(PHELPS_DEATON)

    # Variable Info
    assert ~(model.variable_info["is_sparse"].to_numpy()).any()

    assert (
        model.variable_info["is_state"].to_numpy() == np.array([True, False, False])
    ).all()

    assert (
        model.variable_info["is_continuous"].to_numpy() == np.array([True, False, True])
    ).all()

    # Gridspecs
    wealth_specs = GridSpec(
        kind="linspace",
        specs={"start": 0, "stop": 100, "n_points": N_STATE_GRID_POINTS},
    )

    assert model.gridspecs["wealth"] == wealth_specs

    consumption_specs = GridSpec(
        kind="linspace",
        specs={"start": 0, "stop": 100, "n_points": N_CHOICE_GRID_POINTS},
    )
    assert model.gridspecs["consumption"] == consumption_specs

    assert model.gridspecs["retirement"] == [0, 1]

    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].specs)
    assert (asserted == model.grids["consumption"]).all()

    func = getattr(grids_module, model.gridspecs["wealth"].kind)
    asserted = func(**model.gridspecs["wealth"].specs)
    assert (asserted == model.grids["wealth"]).all()

    assert (model.grids["retirement"] == jax.numpy.array([0, 1])).all()

    # Functions
    assert (
        model.function_info["is_next"].to_numpy()
        == np.array([False, True, False, False])
    ).all()

    assert (
        model.function_info["is_constraint"].to_numpy()
        == np.array([False, False, True, False])
    ).all()

    assert ~model.function_info.loc["utility"].to_numpy().any()
