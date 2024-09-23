# Copyright 2019 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import itertools
import operator
from collections.abc import Sequence

import jax.numpy as jnp
from jax import lax
from jax._src import api, util
from jax._src.typing import Array, ArrayLike
from jax._src.util import safe_zip as zip


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.add, arrs)


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)


def _linear_indices_and_weights(
    coordinate: Array, input_size: int
) -> list[tuple[Array, ArrayLike]]:
    lower = jnp.clip(jnp.floor(coordinate), min=0, max=input_size - 2)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(jnp.int32)
    return [(index, lower_weight), (index + 1, upper_weight)]


@functools.partial(api.jit, static_argnums=(2,))
def _map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    cval: ArrayLike = 0.0,
) -> Array:
    input_arr = jnp.asarray(input)
    coordinate_arrs = [jnp.asarray(c) for c in coordinates]
    cval = jnp.asarray(cval, input_arr.dtype)

    if len(coordinates) != input_arr.ndim:
        raise ValueError(
            "coordinates must be a sequence of length input.ndim, but "
            f"{len(coordinates)} != {input_arr.ndim}"
        )

    index_fixer = lambda index, size: jnp.clip(index, 0, size - 1)
    is_valid = lambda index, size: True
    interp_fun = _linear_indices_and_weights

    valid_1d_interpolations = []
    for coordinate, size in zip(coordinate_arrs, input_arr.shape):
        interp_nodes = interp_fun(coordinate, input_size=size)
        valid_interp = []
        for index, weight in interp_nodes:
            fixed_index = index_fixer(index, size)
            valid = is_valid(index, size)
            valid_interp.append((fixed_index, valid, weight))
        valid_1d_interpolations.append(valid_interp)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, validities, weights = util.unzip3(items)
        if all(valid is True for valid in validities):
            # fast path
            contribution = input_arr[indices]
        else:
            all_valid = functools.reduce(operator.and_, validities)
            contribution = jnp.where(all_valid, input_arr[indices], cval)
        outputs.append(_nonempty_prod(weights) * contribution)  # type: ignore
    result = _nonempty_sum(outputs)
    if jnp.issubdtype(input_arr.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)
    return result.astype(input_arr.dtype)


def map_coordinates(
    input: ArrayLike,
    coordinates: Sequence[ArrayLike],
    cval: ArrayLike = 0.0,
):
    """Map the input array to new coordinates using interpolation.

    JAX implementation of :func:`scipy.ndimage.map_coordinates`

    Given an input array and a set of coordinates, this function returns the
    interpolated values of the input array at those coordinates.

    Args:
      input: N-dimensional input array from which values are interpolated.
      coordinates: length-N sequence of arrays specifying the coordinates
        at which to evaluate the interpolated values
      order: The order of interpolation. JAX supports the following:

        * 0: Nearest-neighbor
        * 1: Linear

      mode: Points outside the boundaries of the input are filled according to the given mode.
        JAX supports one of ``('constant', 'nearest', 'mirror', 'wrap', 'reflect')``. Note the
        ``'wrap'`` mode in JAX behaves as ``'grid-wrap'`` mode in SciPy, and ``'constant'``
        mode in JAX behaves as ``'grid-constant'`` mode in SciPy. This discrepancy was caused
        by a former bug in those modes in SciPy (https://github.com/scipy/scipy/issues/2640),
        which was first fixed in JAX by changing the behavior of the existing modes, and later
        on fixed in SciPy, by adding modes with new names, rather than fixing the existing
        ones, for backwards compatibility reasons. Default is 'constant'.
      cval: Value used for points outside the boundaries of the input if ``mode='constant'``
        Default is 0.0.

    Returns:
      The interpolated values at the specified coordinates.

    Examples:
      >>> input = jnp.arange(12.0).reshape(3, 4)
      >>> input
      Array([[ 0.,  1.,  2.,  3.],
             [ 4.,  5.,  6.,  7.],
             [ 8.,  9., 10., 11.]], dtype=float32)
      >>> coordinates = [jnp.array([0.5, 1.5]),
      ...                jnp.array([1.5, 2.5])]
      >>> jax.scipy.ndimage.map_coordinates(input, coordinates, order=1)
      Array([3.5, 8.5], dtype=float32)

    Note:
      Interpolation near boundaries differs from the scipy function, because JAX
      fixed an outstanding bug; see https://github.com/google/jax/issues/11097.
      This function interprets the ``mode`` argument as documented by SciPy, but
      not as implemented by SciPy.
    """
    return _map_coordinates(input, coordinates, cval)
