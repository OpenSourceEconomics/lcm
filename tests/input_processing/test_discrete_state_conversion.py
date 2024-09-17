from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from lcm import DiscreteGrid
from lcm.input_processing.discrete_state_conversion import (
    _func_depends_on,
    _get_discrete_vars_with_non_index_options,
    _get_index_to_label_func,
    _get_next_func_of_index_var,
    convert_discrete_options_to_indices,
)


@dataclass
class ModelMock:
    """A model mock for testing the process_model function.

    This dataclass has the same attributes as the Model dataclass, but does not perform
    any checks, which helps us to test the process_model function in isolation.

    """

    n_periods: int
    functions: dict[str, Any]
    choices: dict[str, Any]
    states: dict[str, Any]


@pytest.fixture
def model():
    def next_c(a, b):
        return a + b

    return ModelMock(
        n_periods=2,
        functions={
            "next_c": next_c,
        },
        choices={
            "a": DiscreteGrid([0, 1]),
        },
        states={
            "c": DiscreteGrid([1, 10]),
        },
    )


def test_get_index_to_label_func():
    labels = jnp.array([1, 10])
    got = _get_index_to_label_func(labels_array=labels, name="foo")
    assert got(__foo_index__=0) == 1
    assert got(1) == 10


def test_get_discrete_vars_with_non_index_options(model):
    got = _get_discrete_vars_with_non_index_options(model)
    assert got == ["c"]


def test_convert_discrete_options_to_indices(model):
    # add replace method to model mock
    model.replace = lambda **kwargs: ModelMock(**kwargs, n_periods=model.n_periods)

    got, _ = convert_discrete_options_to_indices(model)

    assert "c" not in got.states
    assert "__c_index__" in got.states
    assert "c" in got.functions
    assert_array_equal(got.states["__c_index__"], DiscreteGrid([0, 1]))
    assert got.functions["c"](0) == 1
    assert got.functions["c"](1) == 10


def test_func_depends_on():
    def foo(a, b):
        pass

    assert _func_depends_on(foo, depends_on=["a", "b"])
    assert not _func_depends_on(foo, depends_on=["c"])


def test_get_next_func_of_index_var():
    def next_a(a):
        return a

    got = _get_next_func_of_index_var(next_a, variables=["a"])
    assert got(__a_index__=0) == 0
    assert got(2) == 2
