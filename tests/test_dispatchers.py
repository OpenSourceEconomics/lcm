import itertools

import jax.numpy as jnp
import pytest
from lcm.dispatchers import product_map
from lcm.dispatchers import state_space_map
from numpy.testing import assert_array_almost_equal as aaae


def f(a, b, c):
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


def g(a, b, c, d):
    return f(a, b, c) + jnp.log(d)


def test_product_map_with_all_arguments_mapped():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T

    expected = f(*helper).reshape(10, 7, 5)

    decorated = product_map(f, ["a", "b", "c"])
    calculated = decorated(*grids.values())
    aaae(calculated, expected)


def test_product_map_with_all_arguments_mapped_some_len_one():
    grids = {
        "a": jnp.array([1]),
        "b": jnp.array([2]),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T

    expected = f(*helper).reshape(1, 1, 5)

    decorated = product_map(f, ["a", "b", "c"])
    calculated = decorated(*grids.values())
    aaae(calculated, expected)


def test_product_map_with_all_arguments_mapped_some_scalar():
    grids = {
        "a": 1,
        "b": 2,
        "c": jnp.linspace(1, 5, 5),
    }

    decorated = product_map(f, ["a", "b", "c"])
    error_msg = "vmap got arg 0 of rank 0 but axis to be mapped 0"
    with pytest.raises(ValueError, match=error_msg):
        decorated(*grids.values())


def test_product_map_with_some_arguments_mapped():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": 1,
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(grids["a"], [grids["b"]], grids["c"]))).T

    expected = f(*helper).reshape(10, 5)

    decorated = product_map(f, ["a", "c"])
    calculated = decorated(*grids.values())
    aaae(calculated, expected)


def test_state_space_map_all_arguments_mapped():
    simple_variables = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    complex_grids = {
        "c": jnp.array([7.0, 8, 9, 10]),
        "d": jnp.array([9.0, 10, 11, 12, 13]),
    }

    helper = jnp.array(list(itertools.product(*complex_grids.values()))).T

    complex_variables = {
        "c": helper[0],
        "d": helper[1],
    }

    all_grids = {**simple_variables, **complex_grids}

    helper = jnp.array(list(itertools.product(*all_grids.values()))).T
    expected = g(*helper).reshape(3, 2, 4 * 5)

    decorated = state_space_map(g, list(simple_variables), list(complex_variables))
    calculated = decorated(**simple_variables, **complex_variables)

    aaae(calculated, expected)
