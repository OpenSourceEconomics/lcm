import functools
from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from lcm.argmax import argmax
from lcm.dispatchers import productmap
from lcm.typing import ArgmaxQOverCFunction, MaxQOverCFunction, ParamsDict, Scalar


def get_max_Q_over_c(
    utility_and_feasibility: Callable[..., tuple[Array, Array]],
    continuous_actions_names: tuple[str, ...],
    states_and_discrete_actions_names: tuple[str, ...],
) -> MaxQOverCFunction:
    r"""Get function that maximizes the Q-function over continuous actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  U(x, a) + \beta * \mathbb{E}[V(x', a') | x, a].
    ```

    Fixing a state and discrete action, maximizing over the continuous actions, we get
    the $Q^c$ function:

    ```{math}
    Q^{c}(x, a^d) = \max_{a^c} Q(x, a^d, a^c).
    ```

    The last step is handled by the function returned here.

    Args:
        utility_and_feasibility: A function that takes a state-action combination and
            returns the utility of that combination (scalar) and whether the
            state-action combination is feasible (bool).
        continuous_actions_names: Tuple of action variable names that are continuous.
        states_and_discrete_actions_names: Tuple of state and discrete action variable
            names.

    Returns:
        Function that calculates the maximum of the Q-function over the continuous
        actions. The result corresponds to the Qc-function.

    """
    if continuous_actions_names:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_actions_names,
        )

    @functools.wraps(utility_and_feasibility)
    def max_Q_over_c(vf_arr: Array, params: ParamsDict, **kwargs: Scalar) -> Array:
        u, f = utility_and_feasibility(params=params, vf_arr=vf_arr, **kwargs)
        return u.max(where=f, initial=-jnp.inf)

    return productmap(max_Q_over_c, variables=states_and_discrete_actions_names)


def get_argmax_Q_over_c(
    utility_and_feasibility: Callable[..., tuple[Array, Array]],
    continuous_actions_names: tuple[str, ...],
) -> ArgmaxQOverCFunction:
    r"""Get function that arg-maximizes the Q-function over continuous actions.

    The state-action value function $Q$ is defined as:

    ```{math}
    Q(x, a) =  U(x, a) + \beta * \mathbb{E}[V(x', a') | x, a].
    ```

    Fixing a state and discrete action, arg-maximizing over the continuous actions, we
    get

    ```{math}
    \pi^{c}(x, a^d) = \argmax_{a^c} Q(x, a^d, a^c).
    ```

    The last step is handled by the function returned here.

    Args:
        utility_and_feasibility: A function that takes a state-action combination and
            returns the utility of that combination (scalar) and whether the
            state-action combination is feasible (bool).
        continuous_actions_names: Tuple of action variable names that are continuous.

    Returns:
        Function that calculates the arg-maximum of the Q-function over the continuous
        actions. The result corresponds to the Qc-function.

    """
    if continuous_actions_names:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_actions_names,
        )

    @functools.wraps(utility_and_feasibility)
    def argmax_Q_over_c(
        vf_arr: Array, params: ParamsDict, **kwargs: Scalar
    ) -> tuple[Array, Array]:
        u, f = utility_and_feasibility(params=params, vf_arr=vf_arr, **kwargs)
        return argmax(u, where=f, initial=-jnp.inf)

    return argmax_Q_over_c
