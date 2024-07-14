import jax.numpy as jnp
import lcm.grids as grids_module
import numpy as np
import pandas as pd
import pytest
from lcm import DiscreteGrid, LinspaceGrid, Model
from lcm.mark import StochasticInfo
from lcm.process_model import (
    _get_function_info,
    _get_grids,
    _get_gridspecs,
    _get_stochastic_weight_function,
    _get_variable_info,
    process_model,
)
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from tests.test_models.deterministic import (
    N_GRID_POINTS,
    get_model_config,
)


@pytest.fixture()
def user_model():
    def next_c(a, b):
        return a + b

    return Model(
        n_periods=2,
        functions={
            "next_c": next_c,
        },
        choices={
            "a": DiscreteGrid([0, 1]),
        },
        states={
            "c": DiscreteGrid([0, 1]),
        },
        _skip_checks=True,
    )


def test_get_function_info(user_model):
    got = _get_function_info(user_model)
    exp = pd.DataFrame(
        {
            "is_filter": [False],
            "is_constraint": [False],
            "is_next": [True],
            "is_stochastic_next": [False],
        },
        index=["next_c"],
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
            "is_continuous": [False, False],
            "is_discrete": [True, True],
            "is_stochastic": [False, False],
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
    assert got["a"] == DiscreteGrid([0, 1])
    assert got["c"] == DiscreteGrid([0, 1])


def test_get_grids(user_model):
    variable_info = _get_variable_info(
        user_model,
        function_info=_get_function_info(user_model),
    )
    gridspecs = _get_gridspecs(user_model, variable_info)
    got = _get_grids(gridspecs, variable_info)
    assert_array_equal(got["a"], jnp.array([0, 1]))
    assert_array_equal(got["c"], jnp.array([0, 1]))


def test_process_model_iskhakov_et_al_2017():
    model = process_model(get_model_config("iskhakov_et_al_2017", n_periods=3))

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
    wealth_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=N_GRID_POINTS["wealth"],
    )

    assert model.gridspecs["wealth"] == wealth_specs

    consumption_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=N_GRID_POINTS["consumption"],
    )
    assert model.gridspecs["consumption"] == consumption_specs

    assert model.gridspecs["retirement"] == DiscreteGrid([0, 1])
    assert model.gridspecs["lagged_retirement"] == DiscreteGrid([0, 1])

    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].info._asdict())
    assert (asserted == model.grids["consumption"]).all()

    func = getattr(grids_module, model.gridspecs["wealth"].kind)
    asserted = func(**model.gridspecs["wealth"].info._asdict())
    assert (asserted == model.grids["wealth"]).all()

    assert (model.grids["retirement"] == jnp.array([0, 1])).all()
    assert (model.grids["lagged_retirement"] == jnp.array([0, 1])).all()

    # Functions
    assert (
        model.function_info["is_next"].to_numpy()
        == np.array([False, True, True, False, False, False, False])
    ).all()

    assert (
        model.function_info["is_constraint"].to_numpy()
        == np.array([False, False, False, True, False, False, False])
    ).all()

    assert ~model.function_info.loc["utility"].to_numpy().any()


def test_process_model():
    model = process_model(
        get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )

    # Variable Info
    assert ~(model.variable_info["is_sparse"].to_numpy()).any()

    assert (
        model.variable_info["is_state"].to_numpy() == np.array([False, True, False])
    ).all()

    assert (
        model.variable_info["is_continuous"].to_numpy() == np.array([False, True, True])
    ).all()

    # Gridspecs
    wealth_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=N_GRID_POINTS["wealth"],
    )

    assert model.gridspecs["wealth"] == wealth_specs

    consumption_specs = LinspaceGrid(
        start=1,
        stop=400,
        n_points=N_GRID_POINTS["consumption"],
    )
    assert model.gridspecs["consumption"] == consumption_specs

    assert model.gridspecs["retirement"] == DiscreteGrid([0, 1])

    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].info._asdict())
    assert (asserted == model.grids["consumption"]).all()

    func = getattr(grids_module, model.gridspecs["wealth"].kind)
    asserted = func(**model.gridspecs["wealth"].info._asdict())
    assert (asserted == model.grids["wealth"]).all()

    assert (model.grids["retirement"] == jnp.array([0, 1])).all()

    # Functions
    assert (
        model.function_info["is_next"].to_numpy()
        == np.array([False, True, False, False, False, False, False])
    ).all()

    assert (
        model.function_info["is_constraint"].to_numpy()
        == np.array([False, False, True, False, False, False, False])
    ).all()

    assert ~model.function_info.loc["utility"].to_numpy().any()


def test_get_stochastic_weight_function():
    def raw_func(health, wealth):  # noqa: ARG001
        pass

    raw_func._stochastic_info = StochasticInfo()

    variable_info = pd.DataFrame(
        {"is_discrete": [True, True]},
        index=["health", "wealth"],
    )

    got_function = _get_stochastic_weight_function(
        raw_func,
        name="health",
        variable_info=variable_info,
    )

    params = {"shocks": {"health": np.arange(12).reshape(2, 3, 2)}}

    got = got_function(health=1, wealth=0, params=params)
    expected = np.array([6, 7])
    assert_array_equal(got, expected)


def test_get_stochastic_weight_function_non_state_dependency():
    def raw_func(health, wealth):  # noqa: ARG001
        pass

    raw_func._stochastic_info = StochasticInfo()

    variable_info = pd.DataFrame(
        {"is_discrete": [False, True]},
        index=["health", "wealth"],
    )

    with pytest.raises(ValueError, match="Stochastic variables"):
        _get_stochastic_weight_function(
            raw_func,
            name="health",
            variable_info=variable_info,
        )


def test_variable_info_with_continuous_filter_has_unique_index():
    user_model = get_model_config("iskhakov_et_al_2017", n_periods=3)

    def wealth_filter(wealth):
        return wealth > 200

    user_model.functions["wealth_filter"] = wealth_filter

    function_info = _get_function_info(user_model)
    got = _get_variable_info(
        user_model,
        function_info,
    )
    assert got.index.is_unique
