"""Functions that reduce the conditional continuation values over discrete choices.

By conditional continuation value we mean continuation values conditional on a discrete
choice, i.e. the result of solving the continuous choice problem conditional on the
discrete choice. These are also _conditional_ on a given state.

By reduce we mean calculating the expected maximum of the continuation values, based
on the distribution of utility shocks. Currently we support no shocks. In the future,
we will at least support IID Extreme Value Type 1 shocks.

How we (want to) solve the problem:
-----------------------------------

- No shocks: We take the maximum of the conditional continuation values.

- IID Extreme Value Type 1 shocks: We do a logsum calculation.

"""

from collections.abc import Callable
from functools import partial

import jax
import pandas as pd
from jax import Array

from lcm.typing import ParamsDict, ShockType


def get_solve_discrete_problem(
    *,
    random_utility_shock_type: ShockType,
    variable_info: pd.DataFrame,
    is_last_period: bool,
) -> Callable[[Array], Array]:
    """Get function that computes the expected max. of conditional continuation values.

    The maximum is taken over the discrete choice variables in each state.

    Args:
        random_utility_shock_type: Type of choice shock. Currently only Shock.NONE is
            supported. Work for "extreme_value" is in progress.
        variable_info: DataFrame with information about the variables.
        is_last_period: Whether the function is created for the last period.

    Returns:
        callable: Function that calculates the expected maximum of the conditional
            continuation values. The function depends on `cc_values` (jax.Array), the
            conditional continuation values, and returns the reduced values.

    """
    if is_last_period:
        variable_info = variable_info.query("~is_auxiliary")

    choice_axes = _determine_discrete_choice_axes(variable_info)

    if random_utility_shock_type == ShockType.NONE:
        func = _solve_discrete_problem_no_shocks
    elif random_utility_shock_type == ShockType.EXTREME_VALUE:
        raise NotImplementedError("Extreme value shocks are not yet implemented.")
    else:
        raise ValueError(f"Invalid shock_type: {random_utility_shock_type}.")

    return partial(func, choice_axes=choice_axes)


# ======================================================================================
# Discrete problem with no shocks
# ======================================================================================


def _solve_discrete_problem_no_shocks(
    cc_values: Array,
    choice_axes: tuple[int, ...],
    params: ParamsDict,  # noqa: ARG001
) -> Array:
    """Reduce conditional continuation values over discrete choices.

    Args:
        cc_values (jax.Array): Array with conditional continuation values. For each
            state and discrete choice variable, it has one axis.
        choice_axes (tuple[int, ...]): A tuple of indices representing the axes in the
            value function that correspond to discrete choices. Returns None if there
            are no discrete choice axes.
        params: See `get_solve_discrete_problem`.

    Returns:
        jax.Array: Array with reduced continuation values. Has less dimensions than
            cc_values if choice_axes is not None and is shorter in the first dimension
            if choice_segments is not None.

    """
    return cc_values.max(axis=choice_axes)


# ======================================================================================
# Discrete problem with extreme value shocks
# --------------------------------------------------------------------------------------
# The following is currently *NOT* supported.
# ======================================================================================


def _calculate_emax_extreme_value_shocks(
    values: Array, choice_axes: tuple[int, ...], params: ParamsDict
) -> Array:
    """Aggregate conditional continuation values over discrete choices.

    Args:
        values (jax.Array): Multidimensional jax array with conditional
            continuation values.
        choice_axes (int or tuple): Int or tuple of int, specifying which axes in
            values correspond to the discrete choice variables.
        choice_segments (dict): Dictionary with the entries "segment_ids"
            and "num_segments". segment_ids are a 1d integer array that partitions the
            first dimension of values into choice sets over which we need to aggregate.
            "num_segments" is the number of choice sets.
        params (dict): Params dict that contains the schock_scale if necessary.

    Returns:
        jax.Array: Multidimensional jax array with aggregated continuation
        values. Has less dimensions than values if choice_axes is not None and
        is shorter in the first dimension if choice_segments is not None.

    """
    scale = params["additive_utility_shock"]["scale"]
    return scale * jax.scipy.special.logsumexp(values / scale, axis=choice_axes)


# ======================================================================================
# Auxiliary functions
# ======================================================================================


def _determine_discrete_choice_axes(
    variable_info: pd.DataFrame,
) -> tuple[int, ...]:
    """Get axes of a state-choice-space that correspond to discrete choices.

    Args:
        variable_info: DataFrame with information about the variables.

    Returns:
        A tuple of indices representing the axes' positions in the value function that
        correspond to discrete choices.

    """
    discrete_choice_vars = set(
        variable_info.query("is_choice & is_discrete").index.tolist()
    )
    return tuple(
        i for i, ax in enumerate(variable_info.index) if ax in discrete_choice_vars
    )
