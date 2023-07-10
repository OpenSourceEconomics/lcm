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
        - jax.numpy.ndarray: Array with the same shape as a, except for the dimensions
          specified in axis, which are dropped. The value corresponds to an index
          that can be translated into a tuple of indices using jnp.unravel_index.

    """
    # Preparation
    # ==================================================================================
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
        where = _move_axes_to_back(where, axes=axis)
        where = _flatten_last_n_axes(where, n=len(axis))

    # Compute argmax over last dimension
    # ==================================================================================
    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)
    max_value_mask = a == _max
    if where is not None:
        max_value_mask = max_value_mask & where
    argmax = jnp.argmax(max_value_mask, axis=-1)

    return argmax, _max.reshape(argmax.shape)


def _move_axes_to_back(a, axes):
    """Move specified axes to the back of the array.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        axes (tuple): Axes to move to the back.

    Returns:
        jax.numpy.ndarray: Array a with shifted axes.

    """
    front_axes = sorted(set(range(a.ndim)) - set(axes))
    return a.transpose((*front_axes, *axes))


def _flatten_last_n_axes(a, n):
    """Flatten the last n axes of a to 1 dimension.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        n (int): Number of axes to flatten.

    Returns:
        jax.numpy.ndarray: Array a with flattened last n axes.

    """
    return a.reshape(*a.shape[:-n], -1)


# ======================================================================================
# segment argmax
# ======================================================================================


def segment_argmax(a, segment_ids, num_segments):
    """Calculate a segment argmax over the first axis of a.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        segment_ids (jax.numpy.ndarray): 1d array with segment identifiers. See
            jax.ops.segment_max.
        num_segments (int): Total number of segments. See jax.ops.segment_max.

    Returns:
        jax.numpy.ndarray: Array with shape (num_segments, *a.shape[1:]). The value
            for the k-th segment will be in jnp.arange(segment_ids[k]).

    """
    # Compute segment maximum and bring to the same shape as a
    # ==================================================================================
    segment_max = jax.ops.segment_max(
        data=a,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=True,
    )
    segment_max_expanded = segment_max[segment_ids]

    # Check where the array attains its maximum
    # ==================================================================================
    max_value_mask = a == segment_max_expanded

    # Create index array of argmax indices for each segment (has same shape as a)
    # ==================================================================================
    arange = jnp.arange(a.shape[0])
    reshaped = arange.reshape(-1, *((1,) * (len(a.shape) - 1)))
    segment_argmax_ids = jnp.broadcast_to(reshaped, a.shape)

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
