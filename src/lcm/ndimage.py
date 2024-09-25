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
#
# Modifications made by Tim Mensinger, 2024

import functools
import itertools
import operator
from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array, lax, util


def map_coordinates(
    input: Array,
    coordinates: Sequence[Array],
) -> Array:
    """Map the input array to new coordinates using linear interpolation.

    Modified from JAX implementation of :func:`scipy.ndimage.map_coordinates`.

    Given an input array and a set of coordinates, this function returns the
    interpolated values of the input array at those coordinates. For coordinates outside
    the input array, linear extrapolation is used.

    Args:
      input: N-dimensional input array from which values are interpolated.
      coordinates: length-N sequence of arrays specifying the coordinates
        at which to evaluate the interpolated values

    Returns:
      The interpolated (extrapolated) values at the specified coordinates.

    """
    if len(coordinates) != input.ndim:
        raise ValueError(
            "coordinates must be a sequence of length input.ndim, but "
            f"{len(coordinates)} != {input.ndim}"
        )

    valid_1d_interpolations = []
    for coordinate, size in util.safe_zip(coordinates, input.shape):
        interp_nodes = _linear_indices_and_weights(coordinate, input_size=size)
        valid_1d_interpolations.append(interp_nodes)

    outputs = []
    for items in itertools.product(*valid_1d_interpolations):
        indices, weights = util.unzip2(items)
        contribution = input[indices]
        outputs.append(_nonempty_prod(weights) * contribution)

    result = _nonempty_sum(outputs)

    if jnp.issubdtype(input.dtype, jnp.integer):
        result = _round_half_away_from_zero(result)

    return result.astype(input.dtype)


def _linear_indices_and_weights(
    coordinate: Array, input_size: int
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    lower = jnp.clip(jnp.floor(coordinate), min=0, max=input_size - 2)
    upper_weight = coordinate - lower
    lower_weight = 1 - upper_weight
    index = lower.astype(jnp.int32)
    return (index, lower_weight), (index + 1, upper_weight)


def _nonempty_prod(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.mul, arrs)


def _nonempty_sum(arrs: Sequence[Array]) -> Array:
    return functools.reduce(operator.add, arrs)


def _round_half_away_from_zero(a: Array) -> Array:
    return a if jnp.issubdtype(a.dtype, jnp.integer) else lax.round(a)
