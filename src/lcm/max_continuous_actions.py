import functools
from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from lcm.argmax import argmax
from lcm.dispatchers import productmap
from lcm.typing import ParamsDict


def get_compute_conditional_continuation_value(
    utility_and_feasibility: Callable[..., tuple[Array, Array]],
    continuous_action_variables: tuple[str, ...],
) -> Callable[..., Array]:
    """Get a function that computes the conditional continuation value.

    This function solves the continuous action problem conditional on a state-
    (discrete-)action combination; and is used in the model solution process.

    Args:
        utility_and_feasibility: A function that takes a state-action combination and
            returns the utility of that combination (scalar) and whether the
            state-action combination is feasible (bool).
        continuous_action_variables: Tuple of action variable names that are continuous.

    Returns:
        A function that takes a state-action combination and returns the maximum
        conditional continuation value over the feasible continuous actions.

    """
    if continuous_action_variables:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_action_variables,
        )

    @functools.wraps(utility_and_feasibility)
    def compute_ccv(params: ParamsDict, **kwargs: Array) -> Array:
        u, f = utility_and_feasibility(params=params, **kwargs)
        return u.max(where=f, initial=-jnp.inf)

    return compute_ccv


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
