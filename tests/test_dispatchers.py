import itertools

import jax.numpy as jnp
import pytest
from jax import config
from lcm.dispatchers import product_map
from lcm.dispatchers import state_space_map
from numpy.testing import assert_array_almost_equal as aaae

config.update("jax_enable_x64", True)


def f(a, b, c):
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


def f2(b, a, c):
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


def g(a, b, c, d):
    return f(a, b, c) + jnp.log(d)


# ======================================================================================
# product_map
# ======================================================================================


@pytest.fixture()
def setup_product_map_f():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }
    return grids


@pytest.fixture()
def expected_product_map_f():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T
    expected_result = f(*helper).reshape(10, 7, 5)
    return expected_result


@pytest.fixture()
def setup_product_map_g():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
        "d": jnp.linspace(1, 3, 4),
    }
    return grids


@pytest.fixture()
def expected_product_map_g():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
        "d": jnp.linspace(1, 3, 4),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T
    expected_result = g(*helper).reshape(10, 7, 5, 4)
    return expected_result


@pytest.mark.parametrize(
    "func, args, grids, expected",
    [
        (f, ["a", "b", "c"], "setup_product_map_f", "expected_product_map_f"),
        (g, ["a", "b", "c", "d"], "setup_product_map_g", "expected_product_map_g"),
    ],
)
def test_product_map_with_all_arguments_mapped(func, args, grids, expected, request):
    grids = request.getfixturevalue(grids)
    expected = request.getfixturevalue(expected)

    decorated = product_map(func, args)
    calculated_args = decorated(*grids.values())
    calculated_kwargs = decorated(**grids)

    aaae(calculated_args, expected)
    aaae(calculated_kwargs, expected)


def test_product_map_different_func_order(setup_product_map_f):
    decorated_f = product_map(f, ["a", "b", "c"])
    expected = decorated_f(*setup_product_map_f.values())

    decorated_f2 = product_map(f2, ["a", "b", "c"])
    calculated_f2_kwargs = decorated_f2(**setup_product_map_f)

    aaae(calculated_f2_kwargs, expected)


def test_product_map_change_arg_order(setup_product_map_f, expected_product_map_f):
    expected = jnp.transpose(expected_product_map_f, (1, 0, 2))

    decorated = product_map(f, ["b", "a", "c"])
    calculated = decorated(**setup_product_map_f)

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
    with pytest.raises(ValueError):
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


def test_product_map_with_some_argument_mapped_twice():
    error_msg = "Same argument provided more than once."
    with pytest.raises(ValueError, match=error_msg):
        product_map(f, ["a", "a", "c"])


# ======================================================================================
# state_space_map
# ======================================================================================


@pytest.fixture()
def setup_state_space_map():
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
    return simple_variables, complex_variables


@pytest.fixture()
def expected_state_space_map():
    simple_variables = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    complex_grids = {
        "c": jnp.array([7.0, 8, 9, 10]),
        "d": jnp.array([9.0, 10, 11, 12, 13]),
    }

    all_grids = {**simple_variables, **complex_grids}
    helper = jnp.array(list(itertools.product(*all_grids.values()))).T

    expected_result = g(*helper).reshape(3, 2, 4 * 5)
    return expected_result


def test_state_space_map_all_arguments_mapped(
    setup_state_space_map, expected_state_space_map
):
    simple_variables, complex_variables = setup_state_space_map

    decorated = state_space_map(g, list(simple_variables), list(complex_variables))
    calculated = decorated(**simple_variables, **complex_variables)

    aaae(calculated, expected_state_space_map)


@pytest.mark.parametrize(
    "error_msg, simple_vars, complex_vars",
    [
        (
            "Simple and complex variables overlap.",
            ["a", "b"],
            ["a", "c", "d"],
        ),
        (
            "Same argument provided more than once.",
            ["a", "a", "b"],
            ["c", "d"],
        ),
    ],
)
def test_state_space_map_arguments_overlap(error_msg, simple_vars, complex_vars):
    with pytest.raises(ValueError, match=error_msg):
        state_space_map(g, simple_vars, complex_vars)
