from typing import TypedDict

from jax import Array

# Many JAX functions are designed to work with scalar numerical values. This also
# includes zero dimensional jax arrays.
Scalar = int | float | Array


class SegmentInfo(TypedDict):
    """Information on segments which is passed to `jax.ops.segment_max`.

    - "segment_ids" are a 1d integer jax.Array that partitions the first dimension of
      `data` into segments over which we need to aggregate.

    - "num_segments" is the number of segments.

    The segment_ids are assumed to be sorted.

    """

    segment_ids: Array
    num_segments: int
