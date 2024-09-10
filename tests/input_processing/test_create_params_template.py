from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from lcm.grids import DiscreteGrid
from lcm.input_processing.create_params_template import (
    _create_function_params,
    _create_stochastic_transition_params,
    create_params_template,
)


@dataclass
class ModelMock:
    """A model mock for testing the params creation functions.

    This dataclass has the same attributes as the Model dataclass, but does not perform
    any checks, which helps us to test the params creation functions in isolation.

    """

    n_periods: int | None = None
    functions: dict[str, Any] | None = None
    choices: dict[str, Any] | None = None
    states: dict[str, Any] | None = None


def test_create_params_without_shocks():
    model = ModelMock(
        functions={
            "f": lambda a, b, c: None,  # noqa: ARG005
            "next_b": lambda b: b,
        },
        choices={
            "a": DiscreteGrid([0, 1]),
        },
        states={
            "b": DiscreteGrid([0, 1]),
        },
        n_periods=None,
    )
    got = create_params_template(model)
    assert got == {"beta": np.nan, "f": {"c": np.nan}, "next_b": {}}


def test_create_function_params():
    model = ModelMock(
        functions={
            "f": lambda a, b, c: None,  # noqa: ARG005
        },
        choices={
            "a": None,
        },
        states={
            "b": None,
        },
    )
    got = _create_function_params(model)
    assert got == {"f": {"c": np.nan}}


def test_create_shock_params():
    def next_a(a, _period):
        pass

    variable_info = pd.DataFrame(
        {"is_stochastic": True, "is_state": True, "is_discrete": True},
        index=["a"],
    )

    model = ModelMock(
        n_periods=3,
        functions={"next_a": next_a},
    )

    got = _create_stochastic_transition_params(
        user_model=model,
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

    model = ModelMock(
        functions={"next_a": next_a},
    )

    with pytest.raises(ValueError, match="The following variables are stochastic, but"):
        _create_stochastic_transition_params(
            user_model=model,
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

    model = ModelMock(
        functions={"next_a": next_a},
    )

    with pytest.raises(ValueError, match="Stochastic transition functions can only"):
        _create_stochastic_transition_params(
            user_model=model,
            variable_info=variable_info,
            grids={"a": np.array([1, 2])},
        )
