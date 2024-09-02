import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from lcm.create_params_template import (
    _create_function_params,
    _create_stochastic_transition_params,
    create_params_template,
)


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
    got = create_params_template(
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
    def next_a(a, _period):
        pass

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_discrete": True},
        index=["a"],
    )

    got = _create_stochastic_transition_params(
        model_spec={"functions": {"next_a": next_a}, "n_periods": 3},
        variable_info=variable_info,
        grids={"a": np.array([1, 2])},
    )
    assert_equal(got["a"], np.full((2, 3, 2), np.nan))


def test_create_shock_params_invalid_variable():
    def next_a(a):
        pass

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_discrete": False},
        index=["a"],
    )

    with pytest.raises(ValueError, match="The following variables are stochastic, but"):
        _create_stochastic_transition_params(
            model_spec={"functions": {"next_a": next_a}},
            variable_info=variable_info,
            grids={"a": np.array([1, 2])},
        )


def test_create_shock_params_invalid_dependency():
    def next_a(a, b, _period):
        pass

    variable_info = pd.DataFrame(
        {
            "is_stochastic": [True, False],
            "is_state": [True, False],
            "is_discrete": [True, False],
        },
        index=["a", "b"],
    )

    with pytest.raises(ValueError, match="Stochastic transition functions can only"):
        _create_stochastic_transition_params(
            model_spec={"functions": {"next_a": next_a}},
            variable_info=variable_info,
            grids={"a": np.array([1, 2])},
        )
