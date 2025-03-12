from functools import partial

import jax
import pandas as pd
from jax import Array

from lcm.argmax import argmax
from lcm.typing import (
    DiscreteProblemPolicySolverFunction,
    MaxQcFunction,
    ParamsDict,
    ShockType,
)


def get_max_Qc(
    *,
    random_utility_shock_type: ShockType,
    variable_info: pd.DataFrame,
    is_last_period: bool,
) -> MaxQcFunction:
    r"""Get function that computes the (expected) max. of Qc over discrete actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  U(x, a) + \beta * \mathbb{E}[V(x', a') | x, a].
    ```

    Fixing a state and discrete action, maximizing over the continuous actions, we get
    the $Q^c$ function:

    ```{math}
    Q^{c}(x, a^d) = \max_{a^c} Q(x, a^d, a^c).
    ```

    And maximizing over the discrete actions, we get the value function:

    ```{math}
    V(x) = \max_{a^d} Q^{c}(x, a^d).
    ```

    The last step is handled by the function returned here.

    Args:
        random_utility_shock_type: Type of action shock. Currently only Shock.NONE is
            supported. Work for "extreme_value" is in progress.
        variable_info: DataFrame with information about the variables.
        is_last_period: Whether the function is created for the last period.

    Returns:
        Function that calculates the (expected) maximum of Qc over the discrete actions.
        The maximum corresponds to the value function array.

    """
    if is_last_period:
        variable_info = variable_info.query("~is_auxiliary")

    action_axes = _determine_discrete_action_axes_solution(variable_info)

    if random_utility_shock_type == ShockType.NONE:
        func = _max_Qc_no_shocks
    elif random_utility_shock_type == ShockType.EXTREME_VALUE:
        raise NotImplementedError("Extreme value shocks are not yet implemented.")
    else:
        raise ValueError(f"Invalid shock_type: {random_utility_shock_type}.")

    return partial(func, action_axes=action_axes)


def get_solve_discrete_problem_policy(
    *,
    variable_info: pd.DataFrame,
) -> DiscreteProblemPolicySolverFunction:
    """Return a function that calculates the argmax and max of continuation values.

    The argmax is taken over the discrete action variables in each state.

    Args:
        variable_info (pd.DataFrame): DataFrame with information about the model
            variables.

    Returns:
        callable: Function that calculates the argmax of the conditional continuation
            values. The function depends on:
            - values (jax.Array): Multidimensional jax array with conditional
                continuation values.

    """
    action_axes = _determine_discrete_action_axes_simulation(variable_info)

    def _calculate_discrete_argmax(
        values: Array,
        action_axes: tuple[int, ...],
        params: ParamsDict,  # noqa: ARG001
    ) -> tuple[Array, Array]:
        return argmax(values, axis=action_axes)

    return partial(_calculate_discrete_argmax, action_axes=action_axes)


# ======================================================================================
# Discrete problem with no shocks
# ======================================================================================


def _max_Qc_no_shocks(
    Qc_values: Array,
    action_axes: tuple[int, ...],
    params: ParamsDict,  # noqa: ARG001
) -> Array:
    """Take the maximum of Qc over the discrete actions.

    Args:
        Qc_values: The maximum of the state-action value function (Q) over the
            continuous actions, conditional on the discrete action. This has one axis
            for each state and discrete action variable.
        action_axes: Tuple of indices representing the axes in the value function that
            correspond to discrete actions.
        params: See `get_solve_discrete_problem`.

    Returns:
        The maximum of Qc_values over the discrete action axes.

    """
    return Qc_values.max(axis=action_axes)


# ======================================================================================
# Discrete problem with extreme value shocks
# --------------------------------------------------------------------------------------
# The following is currently *NOT* supported.
# ======================================================================================


def _max_Qc_extreme_value_shocks(
    Qc_values: Array, action_axes: tuple[int, ...], params: ParamsDict
) -> Array:
    """Take the expected maximum of Qc over the discrete actions.

    Args:
        Qc_values: The maximum of the state-action value function (Q) over the
            continuous actions, conditional on the discrete action. This has one axis
            for each state and discrete action variable.
        action_axes: Tuple of indices representing the axes in the value function that
            correspond to discrete actions.
        params: See `get_solve_discrete_problem`.

    Returns:
        The expected maximum of Qc_values over the discrete action axes.

    """
    scale = params["additive_utility_shock"]["scale"]
    return scale * jax.scipy.special.logsumexp(Qc_values / scale, axis=action_axes)


# ======================================================================================
# Auxiliary functions
# ======================================================================================


def _determine_discrete_action_axes_solution(
    variable_info: pd.DataFrame,
) -> tuple[int, ...]:
    """Get axes of state-action-space that correspond to discrete actions in solution.

    Args:
        variable_info: DataFrame with information about the variables.

    Returns:
        A tuple of indices representing the axes' positions in the value function that
        correspond to discrete actions.

    """
    discrete_action_vars = set(
        variable_info.query("is_action & is_discrete").index.tolist()
    )
    return tuple(
        i for i, ax in enumerate(variable_info.index) if ax in discrete_action_vars
    )


def _determine_discrete_action_axes_simulation(
    variable_info: pd.DataFrame,
) -> tuple[int, ...]:
    """Get axes of state-action-space that correspond to discrete actions in simulation.

    Args:
        variable_info: DataFrame with information about the variables.

    Returns:
        A tuple of indices representing the axes' positions in the value function that
        correspond to discrete actions.

    """
    discrete_action_vars = set(
        variable_info.query("is_action & is_discrete").index.tolist()
    )

    # The first dimension corresponds to the simulated states, so add 1.
    return tuple(1 + i for i in range(len(discrete_action_vars)))
