import functools
from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from lcm.argmax import argmax
from lcm.dispatchers import productmap
from lcm.typing import MaxQOverCFunction, ParamsDict


def get_max_Q_over_c(
    utility_and_feasibility: Callable[..., tuple[Array, Array]],
    continuous_action_variables: tuple[str, ...],
) -> MaxQOverCFunction:
    """Get function that maximizes the Q-function over continuous actions.

    This function maximizes the state-action value function (Q-function) over the
    continuous actions. The resulting function depends on the state variables and the
    discrete action variables.

    Args:
        utility_and_feasibility: A function that takes a state-action combination and
            returns the utility of that combination (scalar) and whether the
            state-action combination is feasible (bool).
        continuous_action_variables: Tuple of action variable names that are continuous.

    Returns:
        Function that calculates the maximum of the Q-function over the continuous
        actions. The result corresponds to the Qc-function.

    """
    if continuous_action_variables:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_action_variables,
        )

    @functools.wraps(utility_and_feasibility)
    def max_Q_over_c(vf_arr: Array, params: ParamsDict, **kwargs: Array) -> Array:
        u, f = utility_and_feasibility(params=params, vf_arr=vf_arr, **kwargs)
        return u.max(where=f, initial=-jnp.inf)

    return max_Q_over_c


def get_compute_conditional_continuation_policy(
    utility_and_feasibility: Callable[..., tuple[Array, Array]],
    continuous_action_variables: tuple[str, ...],
) -> Callable[..., tuple[Array, Array]]:
    """Get a function that computes the conditional continuation policy.

    This function solves the continuous action problem conditional on a state-
    (discrete-)action combination; and is used in the model simulation process.

    Args:
        utility_and_feasibility: A function that takes a state-action combination and
            return the utility of that combination (scalar) and whether the state-action
            combination is feasible (bool).
        continuous_action_variables: Tuple of action variable names that are
            continuous.

    Returns:
        A function that takes a state-action combination and returns the optimal policy
        (i.e., that index that maximizes the objective function over feasible states x
        action combinations) and the value of the objective function.

    """
    if continuous_action_variables:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_action_variables,
        )

    @functools.wraps(utility_and_feasibility)
    def compute_ccp(params: ParamsDict, **kwargs: Array) -> tuple[Array, Array]:
        u, f = utility_and_feasibility(params=params, **kwargs)
        _argmax, _max = argmax(u, where=f, initial=-jnp.inf)
        return _argmax, _max

    return compute_ccp
