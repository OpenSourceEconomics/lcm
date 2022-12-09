import numpy as np
import jax.numpy as jnp

from lcm.example_models import PHELPS_DEATON_WITH_FILTERS
from lcm.example_models import PHELPS_DEATON
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS
from lcm.process_model import process_model
from lcm.interfaces import GridSpec
import lcm.grids as grids_module


def test_process_model_with_filters():
    model = process_model(
        PHELPS_DEATON_WITH_FILTERS)
    
    # Variable Info
    assert (model.variable_info["is_sparse"].values == np.array(
        [True, True, False, False])).all()

    assert (model.variable_info["is_state"].values == np.array(
        [True, False, True, False])).all()
    
    assert (model.variable_info["is_continuous"].values == np.array(
        [False, False, True, True])).all()

    # Gridspecs
    type(model.gridspecs["wealth"]) == GridSpec
    type(model.gridspecs["consumption"]) == GridSpec
    type(model.gridspecs["retirement"])  == list
    type(model.gridspecs["lagged_retirement"])  == list

    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].specs)
    assert (asserted == model.grids["consumption"]).all()

    type(model.grids["retirement"])  == jnp.array
    type(model.grids["lagged_retirement"])  == jnp.array
    

    # Functions
    assert (model.function_info["is_next"].values == np.array(
        [False, True, True, False, False])).all()
    assert ~ model.function_info.loc["utility"].values.any() 
    

def test_process_model_base():
    model = process_model(
        PHELPS_DEATON)
    
    # Variable Info
    assert ~(model.variable_info["is_sparse"].values).any()
        
    assert (model.variable_info["is_state"].values == np.array(
        [True, False, False])).all()
    
    assert (model.variable_info["is_continuous"].values == np.array(
        [True, False, True])).all()

    # Grids
    type(model.gridspecs["wealth"]) == GridSpec
    type(model.gridspecs["consumption"]) == GridSpec
    type(model.gridspecs["retirement"])  == list
    
    # Grids
    func = getattr(grids_module, model.gridspecs["consumption"].kind)
    asserted = func(**model.gridspecs["consumption"].specs)
    assert (asserted == model.grids["consumption"]).all()
    type(model.grids["retirement"])  == jnp.array
    
    # Functions
    assert (model.function_info["is_next"].values == np.array(
        [False, True, True, False])).all()

    assert (model.function_info["is_constraint"].values == np.array(
        [False, False, True, False])).all()

    assert ~ model.function_info.loc["utility"].values.any() 
    


