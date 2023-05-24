import jax
import jax.numpy as jnp

# ======================================================================================
# argmax
# ======================================================================================


def argmax(a, axis=None, initial=None, where=None):
    """Compute the argmax of a n-dim array along axis.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        axis (int | tuple | None): Axis along which to compute the argmax. If None, the
            argmax is computed over all axes.
        initial (scalar): The minimum value of an output element. Must be present to
            allow computation on empty slice. See ~numpy.ufunc.reduce for details.
        where (jax.numpy.ndarray): Boolean array of the same shape as a. Only the values
            for which where is True are considered for the argmax.

    Returns:
        jax.numpy.ndarray: Array with the same shape as a, except for the dimensions
            specified in axis, which are dropped. The value corresponds to an index
            that can be translated into a tuple of indices using jnp.unravel_index.

    """
    # Preparation
    # ==================================================================================
    if axis is None:
        axis = tuple(range(a.ndim))
    elif isinstance(axis, int):
        axis = (axis,)

    not_considered = tuple(set(range(a.ndim)) - set(axis))
    new_shape = tuple(a.shape[dim] for dim in not_considered)

    # Move axis over which to compute the argmax to the back and flatten last dims
    # ==================================================================================
    a = _transpose_and_reshape(
        a,
        first_dims=not_considered,
        last_dims=axis,
        shape=new_shape,
    )

    # Do same transformation for where
    # ==================================================================================
    if where is not None:
        where = _transpose_and_reshape(
            where,
            first_dims=not_considered,
            last_dims=axis,
            shape=new_shape,
        )

    # Compute argmax over last dimension
    # ==================================================================================
    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)
    helper = a == _max
    if where is not None:
        helper = helper & where
    argmax = jnp.argmax(helper, axis=-1)

    return argmax, _max.reshape(argmax.shape)


def _transpose_and_reshape(a, first_dims, last_dims, shape):
    transposed = a.transpose((*first_dims, *last_dims))
    return transposed.reshape(*shape, -1)


# ======================================================================================
# segment argmax
# ======================================================================================


def segment_argmax(a, segment_ids, num_segments):
    """Calculate a segment argmax over the first axis of a.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        segment_ids (jax.numpy.ndarray): 1d integer array that partitions the first
        num_segments (int): ...

    Returns:
        jax.numpy.ndarray: Array with shape (num_segments, *a.shape[1:]). The value
            for the k-th segment will be in jnp.arange(segment_ids[k]).

    """
    # Preparation
    # ==================================================================================
    bincount = jnp.bincount(segment_ids, length=num_segments)
    # create index arrays for each segment
    segment_indices = jnp.split(jnp.arange(len(segment_ids)), jnp.cumsum(bincount))[:-1]

    # Compute segment maximum and bring to the same shape as a
    # ==================================================================================
    seg_max = jax.ops.segment_max(
        data=a,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
    )
    seg_max = jnp.repeat(seg_max, bincount, axis=0)

    # Check where the array attains its maximum and create segment lists
    # ==================================================================================
    where_max = a == seg_max

    seg_where_max = [where_max[idx] for idx in segment_indices]
    seg_argmax_id = [_nd_arange(count, shape=a.shape[1:]) for count in bincount]

    # Select argmax indices for each segment
    # ==================================================================================
    seg_argmax = [
        jnp.select(_where_max, _argmax_id)
        for _where_max, _argmax_id in zip(seg_where_max, seg_argmax_id, strict=True)
    ]

    return jnp.stack(seg_argmax)


def _nd_arange(stop, shape):
    """Create an array with shape (stop, *shape) where the first axis is an arange.

    Args:
        stop (int): Stop value for the arange.
        shape (tuple): Shape of the last dimensions of the array.

    Returns:
        jax.numpy.ndarray: Array with shape (stop, *shape) where the first axis is an
            arange.

    """
    arr = jnp.arange(stop).reshape((stop,) + (1,) * len(shape))
    return jnp.broadcast_to(arr, shape=(stop, *shape))
