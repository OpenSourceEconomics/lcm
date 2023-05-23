import jax.numpy as jnp


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

    # Move axis over which to compute the argmax to the back
    # ==================================================================================
    a = a.transpose((*not_considered, *axis))

    # Flatten axis over which to compute the argmax
    # ==================================================================================
    a = a.reshape(*new_shape, -1)

    # Do same transformation for where
    # ==================================================================================
    if where is not None:
        where = where.transpose((*not_considered, *axis))
        where = where.reshape(*new_shape, -1)

    # Compute argmax over last dimension
    # ==================================================================================
    _max = jnp.max(a, axis=-1, keepdims=True, initial=initial, where=where)
    helper = a == _max
    if where is not None:
        helper = helper & where
    argmax = jnp.argmax(helper, axis=-1)

    return argmax, _max.reshape(argmax.shape)


def segment_argmax(a, sement_ids, num_segments):  # noqa: ARG001
    pass
