import jax
import lcm.grids as grids_module
import numpy as np
from lcm.example_models import PHELPS_DEATON
from lcm.example_models import PHELPS_DEATON_WITH_FILTERS
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
    assert isinstance(model.gridspecs["wealth"], GridSpec)
    assert isinstance(model.gridspecs["consumption"], GridSpec)
    assert isinstance(model.gridspecs["retirement"], list)
    assert isinstance(model.gridspecs["lagged_retirement"], list)

    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].specs)
    assert (asserted == model.grids["consumption"]).all()

    assert isinstance(model.grids["retirement"], jax.numpy.ndarray)
    assert isinstance(model.grids["lagged_retirement"], jax.numpy.ndarray)

    # Functions
    assert (
        model.function_info["is_next"].to_numpy()
        == np.array([False, True, False, False, False])
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

    # Grids
    assert isinstance(model.gridspecs["wealth"], GridSpec)
    assert isinstance(model.gridspecs["consumption"], GridSpec)
    assert isinstance(model.gridspecs["retirement"], list)

    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].specs)
    assert (asserted == model.grids["consumption"]).all()
    assert isinstance(model.grids["retirement"], jax.numpy.ndarray)

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
