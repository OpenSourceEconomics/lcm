from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from lcm import DiscreteGrid
from lcm.input_processing.discrete_grid_conversion import (
    DiscreteGridConverter,
    _get_code_to_index_func,
    _get_discrete_vars_with_non_index_codes,
    _get_index_to_code_func,
    convert_arbitrary_codes_to_array_indices,
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
def model(category_class_factory):
    def next_c(a, b):
        return a + b

    return ModelMock(
        n_periods=2,
        functions={
            "next_c": next_c,
        },
        choices={
            "a": DiscreteGrid(category_class_factory([0, 1])),
        },
        states={
            # unsorted indices cannot be treated as indices
            "c": DiscreteGrid(category_class_factory([1, 0])),
        },
    )


@pytest.fixture
def discrete_state_converter_kwargs():
    return {
        "converted_states": ["c"],
        "index_to_code": {"c": _get_index_to_code_func(jnp.array([1, 0]), name="c")},
        "code_to_index": {"c": _get_code_to_index_func(jnp.array([1, 0]), name="c")},
    }


def test_discrete_state_converter_internal_to_params(discrete_state_converter_kwargs):
    expected = {
        "next_c": 1,
    }
    internal_params = {
        "next___c_index__": 1,
    }
    converter = DiscreteGridConverter(**discrete_state_converter_kwargs)
    assert converter.internal_to_params(internal_params) == expected


def test_discrete_state_converter_params_to_internal(discrete_state_converter_kwargs):
    expected = {
        "next___c_index__": 1,
    }
    params = {
        "next_c": 1,
    }
    converter = DiscreteGridConverter(**discrete_state_converter_kwargs)
    assert converter.params_to_internal(params) == expected


def test_discrete_state_converter_internal_to_states(discrete_state_converter_kwargs):
    expected = jnp.array([1, 0])
    internal_states = {
        "__c_index__": jnp.array([0, 1]),
    }
    converter = DiscreteGridConverter(**discrete_state_converter_kwargs)
    assert_array_equal(converter.internal_to_states(internal_states)["c"], expected)


def test_discrete_state_converter_states_to_internal(discrete_state_converter_kwargs):
    expected = jnp.array([0, 1])
    states = {
        "c": jnp.array([1, 0]),
    }
    converter = DiscreteGridConverter(**discrete_state_converter_kwargs)
    assert_array_equal(converter.states_to_internal(states)["__c_index__"], expected)


def test_get_index_to_label_func():
    codes_array = jnp.array([1, 0])
    got = _get_index_to_code_func(codes_array, name="foo")
    assert got(__foo_index__=0) == 1
    assert got(1) == 0


def test_get_code_to_index_func():
    codes_array = jnp.array([1, 0])
    got = _get_code_to_index_func(codes_array, name="foo")
    assert_array_equal(got(foo=codes_array), jnp.arange(2))


def test_get_discrete_vars_with_non_index_codes(model):
    got = _get_discrete_vars_with_non_index_codes(model)
    assert got == ["c"]


def test_convert_discrete_codes_to_indices(model):
    # add replace method to model mock
    model.replace = lambda **kwargs: ModelMock(**kwargs, n_periods=model.n_periods)

    got, _ = convert_arbitrary_codes_to_array_indices(model)

    assert "c" not in got.states
    assert "__c_index__" in got.states
    assert "c" in got.functions
    assert got.states["__c_index__"].categories == ["__cat0_index__", "__cat1_index__"]
    assert got.states["__c_index__"].codes == [0, 1]
    assert got.functions["c"](0) == 1
    assert got.functions["c"](1) == 0
