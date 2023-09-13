import numpy as np
import pandas as pd
from lcm.create_params import (
    _create_function_params,
    _create_shock_params,
    create_params,
)
from numpy.testing import assert_equal


def test_create_params_without_shocks():
    model = {
        "functions": {
            "f": lambda a, b, c: None,  # noqa: ARG005
        },
        "choices": {
            "a": None,
        },
        "states": {
            "b": None,
        },
    }
    got = create_params(
        model,
        variable_info=pd.DataFrame({"is_stochastic": [False]}),
        grids=None,
    )
    assert got == {"beta": np.nan, "f": {"c": np.nan}}


def test_create_function_params():
    model = {
        "functions": {
            "f": lambda a, b, c: None,  # noqa: ARG005
        },
        "choices": {
            "a": None,
        },
        "states": {
            "b": None,
        },
    }
    got = _create_function_params(model)
    assert got == {"f": {"c": np.nan}}


def test_create_shock_params():
    def next_a(a):  # noqa: ARG001
        pass

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_discrete": True},
        index=["a"],
    )

    got = _create_shock_params(
        model={"functions": {"next_a": next_a}},
        variable_info=variable_info,
        grids={"a": np.array([1, 2])},
    )
    assert_equal(got["a"], np.full((2, 2), np.nan))
