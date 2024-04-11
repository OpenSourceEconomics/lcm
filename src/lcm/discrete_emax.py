"""Functions to aggregate conditional continuation values over discrete choices.

By conditional_continuation_value we mean continuation values conditional on a discrete
choice, i.e. the result of solving the continuous choice problem conditional on the
discrete choice.

By aggregate we mean calculating the expected maximum of the continuation values, given
based on the distribution of choice shocks. In the long run we plan to support Three
shock distributions (currently only the first two):

- no shocks -> simply take the maximum of the continuation values
- iid extreme value shocks -> do a logsum calculation
- nested logit shocks -> ???

Notes:
- It is possible that we split the aggregate_conditional_continuation values on one
function per shock type, so we can inspect the signatures of the individual functions.
This will only become clear after implementing a few solvers.
- Hopefully, there will be a better way to do segment_logsumexp soon:
https://github.com/google/jax/issues/6265

"""

from functools import partial

import jax
import jax.numpy as jnp


def get_emax_calculator(
    shock_type,
    variable_info,
    is_last_period,
    choice_segments,
    params,
):
    """Return a function that calculates the expected maximum of continuation values.

    The maximum is taken over the discrete choice variables in each state.

    Args:
        shock_type (str or None): One of None, "extreme_value" and "nested_logit".
        variable_info (pd.DataFrame): DataFrame with information about the variables.
        is_last_period (bool): Whether the function is created for the last period.
        choice_segments (jax.numpy.ndarray): Jax array with the indices of the choice
            segments that indicate which sparse choice variables belong to one state.
        params (dict): Dictionary with model parameters.

    Returns:
        callable: Function that calculates the expected maximum of conditional
            continuation values. The function depends on:
            - values (jax.numpy.ndarray): Multidimensional jax array with conditional
                continuation values.

    """
    if is_last_period:
        variable_info = variable_info.query("~is_auxiliary")
    choice_axes = _determine_discrete_choice_axes(variable_info)

    if shock_type is None:
        func = _calculate_emax_no_shocks
    elif shock_type == "extreme_value":
        func = _calculate_emax_extreme_value_shocks
    elif shock_type == "nested_logit":
        raise ValueError("Nested logit shocks are not yet supported.")
    else:
        raise ValueError(f"Invalid shock_type: {shock_type}.")

    return partial(
        func,
        choice_axes=choice_axes,
        choice_segments=choice_segments,
        params=params,
    )


# ======================================================================================
# Discrete problem with no shocks
# ======================================================================================


def _calculate_emax_no_shocks(
    values,
    choice_axes,
    choice_segments,
    params,  # noqa: ARG001
):
    """Aggregate conditional continuation values over discrete choices.

    Args:
        values (jax.numpy.ndarray): Multidimensional jax array with conditional
            continuation values.
        choice_axes (int or tuple): Int or tuple of int, specifying which axes in
            values correspond to dense choice variables.
        choice_segments (dict): Dictionary with the entries "segment_ids"
            and "num_segments". segment_ids are a 1d integer array that partitions the
            first dimension of values into choice sets over which we need to aggregate.
            "num_segments" is the number of choice sets.
        params (dict): Params dict that contains the schock_scale if necessary.

    Returns:
        jax.numpy.ndarray: Multidimensional jax array with aggregated continuation
            values. Has less dimensions than values if choice_axes is not None and
            is shorter in the first dimension if choice_segments is not None.

    """
    out = values
    if choice_axes is not None:
        out = out.max(axis=choice_axes)
    if choice_segments is not None:
        out = _segment_max_over_first_axis(out, choice_segments)

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
    return jax.ops.segment_max(
        data=a,
        indices_are_sorted=True,
        **segment_info,
    )


# ======================================================================================
# Discrete problem with extreme value shocks
# --------------------------------------------------------------------------------------
# The following is currently *NOT* used in any examples.
# ======================================================================================


def _calculate_emax_extreme_value_shocks(values, choice_axes, choice_segments, params):
    """Aggregate conditional continuation values over discrete choices.

    Args:
        values (jax.numpy.ndarray): Multidimensional jax array with conditional
            continuation values.
        choice_axes (int or tuple): Int or tuple of int, specifying which axes in
            values correspond to dense choice variables.
        choice_segments (dict): Dictionary with the entries "segment_ids"
            and "num_segments". segment_ids are a 1d integer array that partitions the
            first dimension of values into choice sets over which we need to aggregate.
            "num_segments" is the number of choice sets.
        params (dict): Params dict that contains the schock_scale if necessary.

    Returns:
        jax.numpy.ndarray: Multidimensional jax array with aggregated continuation
        values. Has less dimensions than values if choice_axes is not None and
        is shorter in the first dimension if choice_segments is not None.

    """
    scale = params["additive_utility_shock"]["scale"]
    out = values
    if choice_axes is not None:
        out = scale * jax.scipy.special.logsumexp(out / scale, axis=choice_axes)
    if choice_segments is not None:
        out = _segment_extreme_value_emax_over_first_axis(out, scale, choice_segments)

    return out


def _segment_extreme_value_emax_over_first_axis(a, scale, segment_info):
    """Calculate emax under iid extreme value assumption over segments of first axis.

    TODO: Explain in more detail how this function is related to EMAX under IID EV.

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
    return scale * _segment_logsumexp(a / scale, segment_info)


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
    return segmax + jnp.log(summed)


# ======================================================================================
# Auxiliary functions
# ======================================================================================


def _determine_discrete_choice_axes(variable_info):
    """Determine which axes of a state_choice_space correspond to discrete choices.

    Args:
        variable_info (pd.DataFrame): DataFrame with information about the variables.

    Returns:
        tuple[int]: Specifies which axes in a value function correspond to discrete
            choices. If no axes correspond to discrete choices, returns None.

    """
    has_sparse = variable_info["is_sparse"].any()
    dense_vars = variable_info.query(
        "is_dense & ~(is_choice & is_continuous)",
    ).index.tolist()

    axes = ["__sparse__", *dense_vars] if has_sparse else dense_vars

    choice_vars = set(variable_info.query("is_choice").index.tolist())

    choice_indices = tuple(i for i, ax in enumerate(axes) if ax in choice_vars)

    if not choice_indices:
        choice_indices = None

    return choice_indices
