import itertools

import jax.numpy as jnp
import pytest
from lcm.dispatchers import (
    productmap,
    spacemap,
    vmap_1d,
)
from lcm.functools import allow_args, allow_kwargs
from numpy.testing import assert_array_almost_equal as aaae


def f(a, /, *, b, c):
    """Tests that dispatchers can handle positional-only and keyword-only arguments.

    a is positional-only, b and c are keyword-only
    """
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


def f2(b, a, /, *, c):
    """Tests that dispatchers can handle positional-only and keyword-only arguments.

    b and a are positional-only, c is keyword-only
    """
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


def g(a, /, b, *, c, d):
    """Tests that dispatchers can handle positional-only and keyword-only arguments.

    a is positional-only, b is positional-or-keyword, c and d are keyword-only
    """
    return f(a, b=b, c=c) + jnp.log(d)


# ======================================================================================
# productmap
# ======================================================================================


@pytest.fixture()
def setup_productmap_f():
    return {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }


@pytest.fixture()
def expected_productmap_f():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T
    return allow_kwargs(allow_args(f))(*helper).reshape(10, 7, 5)


@pytest.fixture()
def setup_productmap_g():
    return {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
        "d": jnp.linspace(1, 3, 4),
    }


@pytest.fixture()
def expected_productmap_g():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
        "d": jnp.linspace(1, 3, 4),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T
    return allow_kwargs(allow_args(g))(*helper).reshape(10, 7, 5, 4)


@pytest.mark.parametrize(
    ("func", "args", "grids", "expected"),
    [
        (f, ["a", "b", "c"], "setup_productmap_f", "expected_productmap_f"),
        (g, ["a", "b", "c", "d"], "setup_productmap_g", "expected_productmap_g"),
    ],
)
def test_productmap_with_all_arguments_mapped(func, args, grids, expected, request):
    grids = request.getfixturevalue(grids)
    expected = request.getfixturevalue(expected)

    decorated = productmap(func, args)
    calculated_args = decorated(*grids.values())
    calculated_kwargs = decorated(**grids)

    aaae(calculated_args, expected)
    aaae(calculated_kwargs, expected)


def test_productmap_different_func_order(setup_productmap_f):
    decorated_f = productmap(f, ["a", "b", "c"])
    expected = decorated_f(*setup_productmap_f.values())

    decorated_f2 = productmap(f2, ["a", "b", "c"])
    calculated_f2_kwargs = decorated_f2(**setup_productmap_f)

    aaae(calculated_f2_kwargs, expected)


def test_productmap_change_arg_order(setup_productmap_f, expected_productmap_f):
    expected = jnp.transpose(expected_productmap_f, (1, 0, 2))

    decorated = productmap(f, ["b", "a", "c"])
    calculated = decorated(**setup_productmap_f)

    aaae(calculated, expected)


def test_productmap_with_all_arguments_mapped_some_len_one():
    grids = {
        "a": jnp.array([1]),
        "b": jnp.array([2]),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T

    expected = allow_kwargs(allow_args(f))(*helper).reshape(1, 1, 5)

    decorated = productmap(f, ["a", "b", "c"])
    calculated = decorated(*grids.values())
    aaae(calculated, expected)


def test_productmap_with_all_arguments_mapped_some_scalar():
    grids = {
        "a": 1,
        "b": 2,
        "c": jnp.linspace(1, 5, 5),
    }

    decorated = productmap(f, ["a", "b", "c"])
    with pytest.raises(ValueError, match="vmap was requested to map its argument"):
        decorated(*grids.values())


def test_productmap_with_some_arguments_mapped():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": 1,
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(grids["a"], [grids["b"]], grids["c"]))).T

    expected = allow_kwargs(allow_args(f))(*helper).reshape(10, 5)

    decorated = productmap(f, ["a", "c"])
    calculated = decorated(*grids.values())
    aaae(calculated, expected)


def test_productmap_with_some_argument_mapped_twice():
    error_msg = "Same argument provided more than once."
    with pytest.raises(ValueError, match=error_msg):
        productmap(f, ["a", "a", "c"])


# ======================================================================================
# spacemap
# ======================================================================================


@pytest.fixture()
def setup_spacemap():
    value_grid = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    sparse_values = {
        "c": jnp.array([7.0, 8, 9, 10]),
        "d": jnp.array([9.0, 10, 11, 12, 13]),
    }

    helper = jnp.array(list(itertools.product(*sparse_values.values()))).T

    combination_grid = {
        "c": helper[0],
        "d": helper[1],
    }
    return value_grid, combination_grid


@pytest.fixture()
def expected_spacemap():
    value_grid = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    combination_grid = {
        "c": jnp.array([7.0, 8, 9, 10]),
        "d": jnp.array([9.0, 10, 11, 12, 13]),
    }

    all_grids = {**value_grid, **combination_grid}
    helper = jnp.array(list(itertools.product(*all_grids.values()))).T

    return allow_kwargs(allow_args(g))(*helper).reshape(3, 2, 4 * 5)


@pytest.mark.parametrize("put_dense_first", [True, False])
def test_spacemap_all_arguments_mapped(
    setup_spacemap,
    expected_spacemap,
    put_dense_first,
):
    dense_vars, sparse_vars = setup_spacemap

    decorated = spacemap(
        g,
        list(dense_vars),
        list(sparse_vars),
        put_dense_first=put_dense_first,
    )
    calculated = decorated(**dense_vars, **sparse_vars)

    if put_dense_first:
        aaae(calculated, expected_spacemap)
    else:
        aaae(calculated, jnp.transpose(expected_spacemap, axes=(2, 0, 1)))


@pytest.mark.parametrize(
    ("error_msg", "dense_vars", "sparse_vars"),
    [
        (
            "Dense and sparse variables must be disjoint. Overlap: {'a'}",
            ["a", "b"],
            ["a", "c", "d"],
        ),
        (
            "Same argument provided more than once in dense variables: {'a'}",
            ["a", "a", "b"],
            ["c", "d"],
        ),
    ],
)
def test_spacemap_arguments_overlap(error_msg, dense_vars, sparse_vars):
    with pytest.raises(ValueError, match=error_msg):
        spacemap(g, dense_vars, sparse_vars, put_dense_first=True)


# ======================================================================================
# vmap_1d
# ======================================================================================


def test_vmap_1d():
    def func(a, b, c):
        return c * (a + b)

    vmapped = vmap_1d(func, variables=["a", "b"])
    a = jnp.array([1, 2])
    got = vmapped(a=a, b=a, c=-1)
    exp = jnp.array([-2, -4])

    aaae(got, exp)


def test_vmap_1d_error():
    with pytest.raises(ValueError, match="Same argument provided more than once."):
        vmap_1d(None, variables=["a", "a"])
