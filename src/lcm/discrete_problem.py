"""Functions that aggregate the conditional continuation values over discrete choices.

By conditional continuation value we mean continuation values conditional on a discrete
choice, i.e. the result of solving the continuous choice problem conditional on the
discrete choice. These are also _conditional_ on a given state.

By aggregate we mean calculating the expected maximum of the continuation values, based
on the distribution of utility shocks. Currently we support no shocks. In the future,
we will at least support IID Extreme Value Type 1 shocks.

How we (want to) solve the problem:
-----------------------------------

- No shocks: We take the maximum of the conditional continuation values.

- IID Extreme Value Type 1 shocks: We do a logsum calculation.

"""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array

from lcm.typing import SegmentInfo


def get_solve_discrete_problem(
    *,
    shock_type: Literal[None, "extreme_value"],
    variable_info: pd.DataFrame,
    is_last_period: bool,
    choice_segments: SegmentInfo | None,
    params: dict[str, Any],
) -> Callable[[Array], Array]:
    """Get function that computes the expected max. of conditional continuation values.

    The maximum is taken over the discrete choice variables in each state.

    Args:
        shock_type (Literal[None, "extreme_value"]): Type of choice shock. Currently
            only None is supported. Work for "extreme_value" is in progress.
        variable_info (pd.DataFrame): DataFrame with information about the variables.
        is_last_period (bool): Whether the function is created for the last period.
        choice_segments (SegmentInfo): Dictionary with the entries "segment_ids" and
            "num_segments". segment_ids are a 1d integer array that partitions the first
            dimension of values into choice sets over which we need to aggregate.
            "num_segments" is the number of choice sets. The segment_ids are assumed to
            be sorted.
        params (dict): Dictionary with model parameters.

    Returns:
        callable: Function that calculates the expected maximum of the conditional
            continuation values. The function depends on `cc_values` (jax.Array), the
            conditional continuation values, and returns the aggregated values.

    """
    if is_last_period:
        variable_info = variable_info.query("~is_auxiliary")

    choice_axes = _determine_dense_discrete_choice_axes(variable_info)

    if shock_type is None:
        func = _solve_discrete_problem_no_shocks
    elif shock_type == "extreme_value":
        raise NotImplementedError("Extreme value shocks are not yet implemented.")
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


def _solve_discrete_problem_no_shocks(
    cc_values: Array,
    choice_axes: tuple[int, ...] | None,
    choice_segments: SegmentInfo | None,
    params: dict[str, Any],  # noqa: ARG001
) -> Array:
    """Aggregate conditional continuation values over discrete choices.

    Args:
        cc_values (jax.Array): Array with conditional continuation values. Has one
            dimensions per state and discrete choice variable.
        choice_axes (tuple[int, ...]): A tuple of indices representing the axes in the
            value function that correspond to discrete choices. Returns None if there
            are no discrete choice axes.
        choice_segments (SegmentInfo): Dictionary with the entries "segment_ids" and
            "num_segments". segment_ids are a 1d integer array that partitions the first
            dimension of values into choice sets over which we need to aggregate.
            "num_segments" is the number of choice sets. The segment_ids are assumed to
            be sorted.
        params (dict): Dictionary with model parameters.

    Returns:
        jax.Array: Array with aggregated continuation values. Has less dimensions than
            cc_values if choice_axes is not None and is shorter in the first dimension
            if choice_segments is not None.

    """
    out = cc_values
    if choice_axes is not None:
        out = out.max(axis=choice_axes)
    if choice_segments is not None:
        out = _segment_max_over_first_axis(out, choice_segments)

    return out


def _segment_max_over_first_axis(
    data: Array,
    segment_info: SegmentInfo,
) -> Array:
    """Calculate a segment_max over the first axis of data.

    Wrapper around ``jax.ops.segment_max``.

    Args:
        data (jax.Array): Multidimensional jax array.
        segment_info (SegmentInfo): Dictionary with the entries "segment_ids" and
            "num_segments". segment_ids are a 1d integer array that partitions the
            first dimension of `data` into segments over which we need to aggregate.
            "num_segments" is the number of segments. The segment_ids are assumed to be
            sorted.

    Returns:
        jax.Array: An array with shape (num_segments,) + data.shape[1:] representing the
            segment maximums.

    """
    return jax.ops.segment_max(
        data=data,
        indices_are_sorted=True,
        **segment_info,
    )


# ======================================================================================
# Discrete problem with extreme value shocks
# --------------------------------------------------------------------------------------
# The following is currently *NOT* supported.
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


def _determine_dense_discrete_choice_axes(
    variable_info: pd.DataFrame,
) -> tuple[int, ...] | None:
    """Get axes of a state choice space that correspond to dense discrete choices.

    Args:
        variable_info (pd.DataFrame): DataFrame with information about the variables.

    Returns:
        tuple[int, ...] | None: A tuple of indices representing the axes in the value
            function that correspond to discrete choices. Returns None if there are no
            discrete choice axes.

    """
    has_sparse = variable_info["is_sparse"].any()

    # List of dense variables excluding continuous choice variables.
    dense_vars = variable_info.query(
        "is_dense & ~(is_choice & is_continuous)",
    ).index.tolist()

    axes = ["__sparse__", *dense_vars] if has_sparse else dense_vars

    choice_vars = set(variable_info.query("is_choice").index.tolist())

    choice_indices = tuple(i for i, ax in enumerate(axes) if ax in choice_vars)

    # Return None if there are no discrete choice axes, otherwise return the indices.
    return choice_indices if choice_indices else None
