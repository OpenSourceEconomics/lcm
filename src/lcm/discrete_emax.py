"""Functions to aggregate conditional continuation values over discrete choices.

By conditional_continuation_value we mean continuation values conditional on a discrete
choice, i.e. the result of solving the continuous choice problem conditional on
the discrete choice.

By aggregate we mean calculating the expected maximum of the continuation values,
given based on the distribution of choice shocks. In the long run we plan to support
Three shock distributions (currently only the first two):

- no shocks -> simply take the maximum of the continuation values
- iid extreme value shocks -> do a logsum calculation
- nested logit shocks -> ???

Notes:

- It is possible that we split the aggregate_conditional_continuation values on
one function per shock type, so we can inspect the signatures of the individual
functions. This will only become clear after implementing a few solvers.
- Hopefully, there will be a better way to do segment_logsumexp soon:
https://github.com/google/jax/issues/6265

"""
import jax
import jax.numpy as jnp


def calculate_discrete_emax(
    values,
    shock_type,
    choice_axes=None,
    choice_segments=None,
    shock_scale=None,
):
    """aggregate conditional continuation values over discrete choices.

    Args:
        values (jax.numpy.ndarray): Multidimensional jax array with conditional
            continuation values.
        shock_type (str or None): One of None, "extreme_value" and "nesed_logit".
        choice_axes (int or tuple): Int or tuple of int, specifying which axes in
            values correspond to dense choice variables.
        choice_segments (dict): Dictionary with the entries "segment_ids"
            and "num_segments". segment_ids are a 1d integer array that partitions the
            first dimension of values into choice sets over which we need to aggregate.
            "num_segments" is the number of choice sets.
        shock_scale (float): The scale parameter of extreme value choice shocks.

    Returns:
        jax.numpy.ndarray: Multidimensional jax array with aggregated continuation
        values. Has less dimensions than values if choice_axes is not None and
        is shorter in the first dimension if choice_segments is not None.

    """
    out = values
    if shock_type is None:
        if choice_axes is not None:
            out = out.max(axis=choice_axes)
        if choice_segments is not None:
            out = _segment_max_over_first_axis(out, choice_segments)

    elif shock_type == "extreme_value":
        scale = shock_scale
        if choice_axes is not None:
            out = scale * jax.scipy.special.logsumexp(out / scale, axis=choice_axes)
        if choice_segments is not None:
            out = _segment_extreme_value_emax_over_first_axis(
                out, scale, choice_segments
            )
    else:
        raise ValueError("Invalid shock_type: {shock_type}.")

    return out


def _segment_max_over_first_axis(a, segment_info):
    """Calculate a segment_max over the first axis of a.

    Wrapper around ``jax.ops.segment_max``.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        segment_info (dict): Dictionary with the entries "segment_ids"
            and "num_segments". segment_ids are a 1d integer array that partitions the
            first dimension of a. "num_segments" is the number of segments. The
            segment_ids are assumed to be sorted.

    Returns:
        jax.numpy.ndarray

    """
    segmax = jax.ops.segment_max(
        data=a,
        indices_are_sorted=True,
        **segment_info,
    )
    return segmax


def _segment_extreme_value_emax_over_first_axis(a, scale, segment_info):
    """Calculate emax under iid extreme value assumption over segments of first axis.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        scale (float): Scale parameter of the extreme value distribution.
        segment_info (dict): Dictionary with the entries "segment_ids"
            and "num_segments". segment_ids are a 1d integer array that partitions the
            last dimension of a. "num_segments" is the number of segments. The
            segment_ids are assumed to be sorted.

    Returns:
        jax.numpy.ndarray

    """

    emax = scale * _segment_logsumexp(a / scale, segment_info)

    return emax


def _segment_logsumexp(a, segment_info):
    """Calculate a logsumexp over segments of the first axis of a.

    We use the familiar logsumexp trick for numerical stability. See:
    https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/ for details.

    Args:
        a (jax.numpy.ndarray): Multidimensional jax array.
        segment_info (dict): Dictionary with the entries "segment_ids"
            and "num_segments". segment_ids are a 1d integer array that partitions the
            first dimension of a. "num_segments" is the number of segments. The
            segment_ids are assumed to be sorted.

    Returns:
        jax.numpy.ndarray

    """
    segmax = jax.ops.segment_max(
        data=a,
        indices_are_sorted=True,
        **segment_info,
    )

    exp = jnp.exp(a - segmax[segment_info["segment_ids"]])

    summed = jax.ops.segment_sum(
        data=exp,
        indices_are_sorted=True,
        **segment_info,
    )
    out = segmax + jnp.log(summed)
    return out
