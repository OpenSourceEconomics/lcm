from functools import partial

import jax.numpy as jnp
import jax.scipy.ndimage
import numpy as np
import pytest
import scipy.ndimage
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

import lcm.ndimage

jax_map_coordinates = partial(jax.scipy.ndimage.map_coordinates, order=1, cval=0)
scipy_map_coordinates = partial(scipy.ndimage.map_coordinates, order=1, cval=0)
lcm_map_coordinates = lcm.ndimage.map_coordinates

JAX_IMPLEMENTATIONS = [jax_map_coordinates, lcm_map_coordinates]


TEST_SHAPES = [
    (5,),
    (3, 4),
    (3, 4, 5),
]

TEST_COORDINATES_SHAPES = [
    (7,),
    (2, 3, 4),
]


def _make_test_data(shape, coordinates_shape, dtype):
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    c = [
        (size - 1) * np.random.rand(*coordinates_shape).astype(dtype) for size in shape
    ]
    return x, c


@pytest.mark.parametrize("map_coordinates", JAX_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("coordinates_shape", TEST_COORDINATES_SHAPES)
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_map_coordinates_against_scipy(
    map_coordinates, shape, coordinates_shape, dtype
):
    """Test that all libraries implement same behavior with integer input."""
    x, c = _make_test_data(shape, coordinates_shape, dtype=dtype)

    x_jax = jnp.asarray(x)
    c_jax = [jnp.asarray(c_i) for c_i in c]

    expected = scipy_map_coordinates(x, c)
    got = map_coordinates(x_jax, c_jax)

    assert_array_almost_equal(got, expected, decimal=14)


@pytest.mark.parametrize("map_coordinates", JAX_IMPLEMENTATIONS)
def test_map_coordinates_round_half_integer_input(map_coordinates):
    """Test that all libraries implement same rounding behavior with integer input."""
    x = np.arange(-5, 5, dtype=np.int64)
    c = np.array([[0.5, 1.5, 2.5, 6.5, 8.5]])

    x_jax = jnp.asarray(x)
    c_jax = [jnp.asarray(c_i) for c_i in c]

    expected = scipy_map_coordinates(x, c)
    got = map_coordinates(x_jax, c_jax)

    assert_array_equal(got, expected)


@pytest.mark.parametrize("map_coordinates", JAX_IMPLEMENTATIONS)
def test_map_coordinates_round_half_float_input(map_coordinates):
    """Test that all libraries implement same rounding behavior with float input."""
    x = np.arange(-5, 5, dtype=np.float64)
    c = np.array([[0.5, 1.5, 2.5, 6.5, 8.5]])

    x_jax = jnp.asarray(x)
    c_jax = [jnp.asarray(c_i) for c_i in c]

    expected = scipy_map_coordinates(x, c)
    got = map_coordinates(x_jax, c_jax)

    assert_array_equal(got, expected)


@pytest.mark.parametrize("map_coordinates", JAX_IMPLEMENTATIONS)
def test_gradients(map_coordinates):
    """Test that JAX based implementations exhibit same gradient behavior."""
    x = jnp.arange(9.0)
    border = 3  # square root of 9, as we are considering a parabola on x.

    def f(step):
        coordinates = x + step
        shifted = map_coordinates(x, [coordinates])
        return ((x - shifted) ** 2)[border:-border].mean()

    # Gradient of f(step) is 2 * step
    assert_allclose(jax.grad(f)(0.5), 1.0)
    assert_allclose(jax.grad(f)(1.0), 2.0)


def test_extrapolation():
    x = jnp.arange(3.0)
    c = [jnp.array([-2.0, -1.0, 5.0, 10.0])]

    got = lcm.ndimage.map_coordinates(x, c)
    expected = c[0]

    assert_array_equal(got, expected)
