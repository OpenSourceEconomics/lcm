import jax
import lcm.grids as grids_module
import numpy as np
from lcm.example_models import N_GRID_POINTS, PHELPS_DEATON, PHELPS_DEATON_WITH_FILTERS
from lcm.interfaces import GridSpec
from lcm.process_model import process_model


def test_process_model_with_filters():
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
        specs={"start": 0, "stop": 100, "n_points": N_GRID_POINTS},
    )

    assert model.gridspecs["wealth"] == wealth_specs

    consumption_specs = GridSpec(
        kind="linspace",
        specs={"start": 1, "stop": 100, "n_points": N_GRID_POINTS},
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


def test_process_model_base():
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
        specs={"start": 0, "stop": 100, "n_points": N_GRID_POINTS},
    )

    assert model.gridspecs["wealth"] == wealth_specs

    consumption_specs = GridSpec(
        kind="linspace",
        specs={"start": 0, "stop": 100, "n_points": N_GRID_POINTS},
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
