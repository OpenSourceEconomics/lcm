import jax.numpy as jnp
from jax import jit
from lcm.argmax import argmax, segment_argmax
from numpy.testing import assert_array_equal

# Test jitted functions
# ======================================================================================
jitted_segment_argmax = jit(segment_argmax, static_argnums=2)
jitted_argmax = jit(argmax, static_argnums=[1, 2, 3])


# ======================================================================================
# argmax
# ======================================================================================


def test_argmax_1d_with_mask():
    a = jnp.arange(10)
    mask = jnp.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    _argmax, _max = jitted_argmax(a, where=mask, initial=-1)
    assert _argmax == 4
    assert _max == 4


def test_argmax_2d_with_mask():
    a = jnp.arange(10).reshape(2, 5)
    mask = jnp.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=bool).reshape(a.shape)

    _argmax, _max = jitted_argmax(a, axis=None, where=mask, initial=-1)
    assert _argmax == 4
    assert _max == 4

    _argmax, _max = jitted_argmax(a, axis=0, where=mask, initial=-1)
    assert_array_equal(_argmax, jnp.array([0, 0, 0, 0, 0]))
    assert_array_equal(_max, jnp.array([0, -1, -1, 3, 4]))

    _argmax, _max = jitted_argmax(a, axis=1, where=mask, initial=-1)
    assert_array_equal(_argmax, jnp.array([4, 0]))
    assert_array_equal(_max, jnp.array([4, -1]))


def test_argmax_1d_no_mask():
    a = jnp.arange(10)
    _argmax, _max = jitted_argmax(a)
    assert _argmax == 9
    assert _max == 9


def test_argmax_2d_no_mask():
    a = jnp.arange(10).reshape(2, 5)

    _argmax, _max = jitted_argmax(a, axis=None)
    assert _argmax == 9
    assert _max == 9

    _argmax, _max = jitted_argmax(a, axis=0)
    assert_array_equal(_argmax, jnp.array([1, 1, 1, 1, 1]))
    assert_array_equal(_max, jnp.array([5, 6, 7, 8, 9]))

    _argmax, _max = jitted_argmax(a, axis=1)
    assert_array_equal(_argmax, jnp.array([4, 4]))
    assert_array_equal(_max, jnp.array([4, 9]))

    _argmax, _max = jitted_argmax(a, axis=(0, 1))
    assert _argmax == 9
    assert _max == 9


def test_argmax_3d_no_mask():
    a = jnp.arange(24).reshape(2, 3, 4)

    _argmax, _max = jitted_argmax(a, axis=None)
    assert _argmax == 23
    assert _max == 23

    _argmax, _max = jitted_argmax(a, axis=0)
    assert_array_equal(_argmax, jnp.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))
    assert_array_equal(
        _max,
        jnp.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]),
    )

    _argmax, _max = jitted_argmax(a, axis=1)
    assert_array_equal(_argmax, jnp.array([[2, 2, 2, 2], [2, 2, 2, 2]]))
    assert_array_equal(_max, jnp.array([[8, 9, 10, 11], [20, 21, 22, 23]]))

    _argmax, _max = jitted_argmax(a, axis=2)
    assert_array_equal(_argmax, jnp.array([[3, 3, 3], [3, 3, 3]]))
    assert_array_equal(_max, jnp.array([[3, 7, 11], [15, 19, 23]]))

    _argmax, _max = jitted_argmax(a, axis=(0, 1))
    assert_array_equal(_argmax, jnp.array([5, 5, 5, 5]))
    assert_array_equal(_max, jnp.array([20, 21, 22, 23]))

    _argmax, _max = jitted_argmax(a, axis=(0, 2))
    assert_array_equal(_argmax, jnp.array([7, 7, 7]))
    assert_array_equal(_max, jnp.array([15, 19, 23]))

    _argmax, _max = jitted_argmax(a, axis=(1, 2))
    assert_array_equal(_argmax, jnp.array([11, 11]))
    assert_array_equal(_max, jnp.array([11, 23]))


# ======================================================================================
# segment argmax
# ======================================================================================


def test_segment_argmax_1d():
    a = jnp.arange(10)
    segment_ids = jnp.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    _argmax, _max = jitted_segment_argmax(a, segment_ids, num_segments=3)
    assert_array_equal(_argmax, jnp.array([2, 4, 9]))
    assert_array_equal(_max, jnp.array([2, 4, 9]))


def test_segment_argmax_2d():
    a = jnp.arange(10).reshape(5, 2)
    segment_ids = jnp.array([0, 0, 0, 1, 1])
    _argmax, _max = jitted_segment_argmax(a, segment_ids, num_segments=2)
    assert_array_equal(_argmax, jnp.array([[2, 2], [4, 4]]))
    assert_array_equal(_max, jnp.array([[4, 5], [8, 9]]))


def test_segment_argmax_3d():
    a = jnp.array(
        [
            [[0, 5], [3, 0]],
            [[1, 2], [0, 0]],
            [[0, 0], [0, 0]],
        ],
    )
    segment_ids = jnp.array([0, 0, 1])
    _argmax, _max = jitted_segment_argmax(a, segment_ids, num_segments=2)
    assert_array_equal(_argmax, jnp.array([[[1, 0], [0, 1]], [[2, 2], [2, 2]]]))
    assert_array_equal(_max, jnp.array([[[1, 5], [3, 0]], [[0, 0], [0, 0]]]))
