import jax.numpy as jnp
import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON, PHELPS_DEATON_WITH_FILTERS
from pybaum import tree_map

MODELS = {
    "simple": PHELPS_DEATON,
    "with_filters": PHELPS_DEATON_WITH_FILTERS,
}


@pytest.mark.parametrize("user_model", list(MODELS.values()), ids=list(MODELS))
def test_get_lcm_function_with_solve_target(user_model):
    solve_model, params_template = get_lcm_function(model=user_model)

    params = tree_map(lambda _: 0.2, params_template)

    solve_model(params)


@pytest.mark.parametrize("user_model", list(MODELS.values()), ids=list(MODELS))
def test_get_lcm_function_with_simulation_target(user_model):
    # solve model
    solve_model, params_template = get_lcm_function(model=user_model)
    params = tree_map(lambda _: 0.2, params_template)
    vf_arr_list = solve_model(params)

    # simulate using solution
    simulate_model, _ = get_lcm_function(model=PHELPS_DEATON, targets="simulation")

    with pytest.raises(NotImplementedError):
        simulate_model(
            params,
            vf_arr_list=vf_arr_list,
            initial_states={"wealth": jnp.array([10.0, 20.0])},
        )
