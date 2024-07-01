from typing import Annotated, Literal, TypedDict

from jax import Array

# Many JAX functions are designed to work with scalar numerical values. This also
# includes zero dimensional jax arrays.
Scalar = int | float | Array

# We only allow primitive types for scalar user inputs to reduce complexity.
ScalarUserInput = int | float

DiscreteLabels = Annotated[list[int], "Int range starting from 0 with increments of 1"]


class SegmentInfo(TypedDict):
    """Information on segments which is passed to `jax.ops.segment_max`.

    - "segment_ids" are a 1d integer jax.Array that partitions the first dimension of
      `data` into segments over which we need to aggregate.

    - "num_segments" is the number of segments.

    The segment_ids are assumed to be sorted.

    """

    segment_ids: Array
    num_segments: int


class MapCoordinatesOptions(TypedDict):
    """Options passed to  `jax.scipy.ndimage.map_coordinates`.

    From the JAX documentation (as of 2024-06-16):

    - "order": The order of interpolation. JAX supports the following:
      - 0: Nearest-neighbor
      - 1: Linear

    - "mode": Points outside the boundaries of the input are filled according to the
      given mode. JAX supports one of ('constant', 'nearest', 'mirror', 'wrap',
      'reflect'). Note the 'wrap' mode in JAX behaves as 'grid-wrap' mode in SciPy, and
      'constant' mode in JAX behaves as 'grid-constant' mode in SciPy. This discrepancy
      was caused by a former bug in those modes in SciPy (scipy/scipy#2640), which was
      first fixed in JAX by changing the behavior of the existing modes, and later on
      fixed in SciPy, by adding modes with new names, rather than fixing the existing
      ones, for backwards compatibility reasons.

    - "cval": Value used for points outside the boundaries of the input if
      mode='constant'.

    """

    order: Literal[0, 1]
    mode: Literal["constant", "nearest", "mirror", "wrap", "reflect"]
    cval: Scalar
