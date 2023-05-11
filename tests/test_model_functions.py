import pandas as pd
import pytest
from lcm.interfaces import GridSpec, Model, SpaceInfo
from lcm.model_functions import (
    get_combined_constraint,
    get_utility_and_feasibility_function,
)


@pytest.mark.skip(reason="Not ready yet.")
def test_get_utility_and_feasibility_function():
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

    grid_specs = {"start": 0, "stop": 1, "n_points": 3}

    space_info = SpaceInfo(
        axis_names=["a"],
        lookup_info={},
        interpolation_info={
            "a": GridSpec(kind="linspace", specs=grid_specs),
        },
        indexer_infos=[],
    )

    model = Model(
        grids=None,
        gridspecs=None,
        variable_info=None,
        functions={"f": f, "g": g, "h": h},
        function_info=function_info,
        params=None,
        shocks=None,
        n_periods=None,
    )

    get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        data_name="values_name",
        interpolation_options=None,
        is_last_period=False,
    )


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
    model = Model(
        grids=None,
        gridspecs=None,
        variable_info=None,
        functions={"f": f, "g": g, "h": h},
        function_info=function_info,
        params=None,
        shocks=None,
        n_periods=None,
    )
    combined_constraint = get_combined_constraint(model)
    assert not combined_constraint()
