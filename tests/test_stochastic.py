from lcm.entry_point import (
    get_lcm_function,
)
from lcm.example_models_stochastic import MODEL
from pybaum import tree_map

# ======================================================================================
# Solve
# ======================================================================================


def test_get_lcm_function_with_solve_target():
    solve_model, params_template = get_lcm_function(model=MODEL)

    params = tree_map(lambda _: 0.2, params_template)

    solve_model(params)
