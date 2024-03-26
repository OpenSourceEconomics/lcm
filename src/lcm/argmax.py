import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

# ======================================================================================
# argmax
# ======================================================================================


def argmax(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Compute the argmax of an n-dim array along axis.

    If multiple maxima exist, the first index will be selected.

    Args:
        a (ArrayLike): Multidimensional array.
        axis (int | tuple | None): Axis along which to compute the argmax. If None, the
            argmax is computed over all axes.
        initial (ArrayLike): The minimum value of an output element. Must be present to
            allow computation on empty slice. See ~numpy.ufunc.reduce for details.
        where (ArrayLike): Elements to compare for the maximum. See ~numpy.ufunc.reduce
            for details.

    Returns:
        - Array: The argmax indices. Array with the same shape as a, except for
            the dimensions specified in axis, which are dropped. The value corresponds
            to an index that can be translated into a tuple of indices using
            jnp.unravel_index.

        - Array: The corresponding maximum values.

    """
    # Preparation
    # ==================================================================================
    a = jnp.asarray(a)

    if axis is None:
        axis = tuple(range(a.ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    # Move axis over which to compute the argmax to the back and flatten last dims
    # ==================================================================================
    a = _move_axes_to_back(a, axes=axis)
    a = _flatten_last_n_axes(a, n=len(axis))

    # Do same transformation for where
    # ==================================================================================
    if where is not None:
        where = jnp.asarray(where)
        where = _move_axes_to_back(where, axes=axis)
        where = _flatten_last_n_axes(where, n=len(axis))

    # Compute argmax over last dimension
    # ----------------------------------------------------------------------------------
    # Note: If multiple maxima exist, this approach will select the first index.
    # ==================================================================================
    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)
    max_value_mask = a == _max
    if where is not None:
        max_value_mask = jnp.logical_and(max_value_mask, where)
    argmax = jnp.argmax(max_value_mask, axis=-1)

    return argmax, _max.reshape(argmax.shape)


def _move_axes_to_back(a: Array, axes: tuple[int, ...]) -> Array:
    """Move specified axes to the back of the array.

    Args:
        a (Array): Multidimensional jax array.
        axes (tuple): Axes to move to the back.

    Returns:
        jax.numpy.ndarray: Array a with shifted axes.

    """
    front_axes = sorted(set(range(a.ndim)) - set(axes))
    return a.transpose((*front_axes, *axes))


def _flatten_last_n_axes(a: Array, n: int) -> Array:
    """Flatten the last n axes of a to 1 dimension.

    Args:
        a (Array): Multidimensional jax array.
        n (int): Number of axes to flatten.

    Returns:
        jax.numpy.ndarray: Array a with flattened last n axes.

    """
    return a.reshape(*a.shape[:-n], -1)


# ======================================================================================
# segment argmax
# ======================================================================================


def segment_argmax(
    data: ArrayLike,
    segment_ids: ArrayLike,
    num_segments: int,
) -> tuple[Array, Array]:
    """Computes the maximum within segments of an array over the first axis of data.

    See `jax.ops.segment_max` for reference. If multiple maxima exist, the last index
    will be selected.

    Args:
        data (ArrayLike): Multidimensional array.
        segment_ids (ArrayLike): An array with integer dtype that indicates the segments
            of data (along its leading axis) to be reduced. Values can be repeated and
            need not be sorted. Values outside of the range [0, num_segments) are
            dropped and do not contribute to the result.
        num_segments (int): An int with nonnegative value indicating the number of
            segments. The default is set to be the minimum number of segments that would
            support all indices in segment_ids, calculated as max(segment_ids) + 1.
            Since num_segments determines the size of the output, a static value must be
            provided to use segment_max in a JIT-compiled function.

    Returns:
        - Array: Array with shape (num_segments, *data.shape[1:]). The value
            for the k-th segment will be in jnp.arange(segment_ids[k]).

        - Array: Array with shape (num_segments, *data.shape[1:]). The maximum
            value for the k-th segment.

    """
    # Preparation
    # ==================================================================================
    data = jnp.asarray(data)
    segment_ids = jnp.asarray(segment_ids)

    # Compute segment maximum and bring to the same shape as data
    # ==================================================================================
    segment_max = jax.ops.segment_max(
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
    )
    segment_max_expanded = segment_max[segment_ids]

    # Check where the array attains its maximum
    # ==================================================================================
    max_value_mask = data == segment_max_expanded

    # Create index array of argmax indices for each segment (has same shape as data)
    # ==================================================================================
    arange = jnp.arange(data.shape[0])
    reshaped = arange.reshape(-1, *([1] * (data.ndim - 1)))
    segment_argmax_ids = jnp.broadcast_to(reshaped, data.shape)

    # Set indices to zero that do not correspond to a maximum
    # ==================================================================================
    max_value_indices = max_value_mask * segment_argmax_ids

    # Select argmax indices for each segment
    # ----------------------------------------------------------------------------------
    # Note: If multiple maxima exist, this approach will select the last index.
    # ==================================================================================
    segment_argmax = jax.ops.segment_max(
        data=max_value_indices,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
    )

    return segment_argmax, segment_max
