import pandas as pd
from lcm.interfaces import Model
from lcm.model_functions import get_combined_constraint


def test_get_combined_constraint():
    def f():
        return True

    def g():
        return False

    def h():
        return None

    function_info = pd.DataFrame(
        {"is_constraint": [True, True, False]},
        index=["f", "g", "h"],
    )
    functions = [f, g, h]
    model = Model(
        grids=None,
        gridspecs=None,
        variable_info=None,
        functions=functions,
        function_info=function_info,
        params=None,
        shocks=None,
        n_periods=None,
    )
    combined_constraint = get_combined_constraint(model)
    assert not combined_constraint()
