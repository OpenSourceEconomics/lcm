from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array

from lcm.dispatchers import productmap
from lcm.function_representation import get_value_function_representation
from lcm.functools import get_union_of_arguments
from lcm.interfaces import InternalModel, StateSpaceInfo
from lcm.next_state import get_next_state_function, get_next_stochastic_weights_function
from lcm.typing import InternalUserFunction, ParamsDict, Scalar, Target


def get_Q_and_F(
    model: InternalModel,
    next_state_space_info: StateSpaceInfo,
    period: int,
) -> Callable[..., tuple[Array, Array]]:
    """Get the state-action (Q) and feasibility (F) function for a given period.

    Args:
        model: The internal model object.
        next_state_space_info: The state space information of the next period.
        period: The current period.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for the given period.

    """
    is_last_period = period == model.n_periods - 1

    if is_last_period:
        Q_and_F = get_Q_and_F_terminal(model, period=period)
    else:
        Q_and_F = get_Q_and_F_non_terminal(
            model, next_state_space_info=next_state_space_info, period=period
        )

    return Q_and_F


def get_Q_and_F_non_terminal(
    model: InternalModel,
    next_state_space_info: StateSpaceInfo,
    period: int,
) -> Callable[..., tuple[Array, Array]]:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    Args:
        model: The internal model object.
        next_state_space_info: The state space information of the next period.
        period: The current period.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    stochastic_variables = model.variable_info.query("is_stochastic").index.tolist()

    # ----------------------------------------------------------------------------------
    # Generate dynamic functions
    # ----------------------------------------------------------------------------------

    # Function required to calculate instantaneous utility and feasibility
    U_and_F = _get_U_and_F(model)

    # Functions required to calculate the expected continuation values
    state_transition = get_next_state_function(model, target=Target.SOLVE)
    next_stochastic_states_weights = get_next_stochastic_weights_function(model)
    joint_weights_from_marginals = _get_joint_weights_function(stochastic_variables)
    _scalar_next_V = get_value_function_representation(next_state_space_info)
    next_V = productmap(
        _scalar_next_V,
        variables=tuple(f"next_{var}" for var in stochastic_variables),
    )

    # ----------------------------------------------------------------------------------
    # Create the state-action value and feasibility function
    # ----------------------------------------------------------------------------------
    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [U_and_F, state_transition, next_stochastic_states_weights],
        include={"params", "next_V_arr"},
        exclude={"_period"},
    )

    @with_signature(args=arg_names_of_Q_and_F)
    def Q_and_F(
        params: ParamsDict, next_V_arr: Array, **states_and_actions: Scalar
    ) -> tuple[Scalar, Scalar]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            params: The parameters.
            next_V_arr: The next period's value function array.
            **states_and_actions: The current states and actions.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        # ------------------------------------------------------------------------------
        # Calculate the expected continuation values
        # ------------------------------------------------------------------------------
        next_states = state_transition(
            **states_and_actions,
            _period=period,
            params=params,
        )

        marginal_next_stochastic_states_weights = next_stochastic_states_weights(
            **states_and_actions,
            _period=period,
            params=params,
        )

        joint_next_stochastic_states_weights = joint_weights_from_marginals(
            **marginal_next_stochastic_states_weights
        )

        # As we productmap'd the value function over the stochastic variables, the
        # resulting next value function gets a new dimension for each stochastic
        # variable.
        next_V_at_stochastic_states_arr = next_V(**next_states, next_V_arr=next_V_arr)

        # We then take the weighted average of the next value function at the stochastic
        # states to get the expected next value function.
        next_V_expected_arr = jnp.average(
            next_V_at_stochastic_states_arr,
            weights=joint_next_stochastic_states_weights,
        )

        # ------------------------------------------------------------------------------
        # Calculate the instantaneous utility and feasibility
        # ------------------------------------------------------------------------------
        U_arr, F_arr = U_and_F(
            **states_and_actions,
            _period=period,
            params=params,
        )

        Q_arr = U_arr + params["beta"] * next_V_expected_arr

        return Q_arr, F_arr

    return Q_and_F


def get_Q_and_F_terminal(
    model: InternalModel,
    period: int,
) -> Callable[..., tuple[Array, Array]]:
    """Get the state-action (Q) and feasibility (F) function for the terminal period.

    Args:
        model: The internal model object.
        period: The current period.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for the terminal period.

    """
    U_and_F = _get_U_and_F(model)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        [U_and_F],
        # While the terminal period does not depend on the value function array, we
        # include it in the signature, such that we can treat all periods uniformly
        # during the solution and simulation.
        include={"params", "next_V_arr"},
        exclude={"_period"},
    )

    @with_signature(args=arg_names_of_Q_and_F)
    def Q_and_F(
        params: ParamsDict,
        next_V_arr: Array,  # noqa: ARG001
        **states_and_actions: Scalar,
    ) -> tuple[Scalar, Scalar]:
        """Calculate the state-action values and feasibilities for the terminal period.

        Args:
            params: The parameters.
            next_V_arr: The next period's value function array (unused here).
            **states_and_actions: The current states and actions.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        return U_and_F(
            **states_and_actions,
            _period=period,
            params=params,
        )

    return Q_and_F


# ======================================================================================
# Helper functions
# ======================================================================================


def _get_arg_names_of_Q_and_F(
    deps: list[Callable[..., Any]],
    include: set[str] = set(),  # noqa: B006
    exclude: set[str] = set(),  # noqa: B006
) -> list[str]:
    """Get the argument names of the dependencies.

    Args:
        deps: List of dependencies.
        include: Set of argument names to include.
        exclude: Set of argument names to exclude.

    Returns:
        The union of the argument names in deps and include, except for those in
        exclude.

    """
    deps_arg_names = get_union_of_arguments(deps)
    return list(include | deps_arg_names - exclude)


def _get_joint_weights_function(
    stochastic_variables: list[str],
) -> Callable[..., Array]:
    """Get function that calculates the joint weights.

    This function takes the weights of the individual stochastic variables and
    multiplies them together to get the joint weights on the product space of the
    stochastic variables.

    Args:
        stochastic_variables: List of stochastic variables.

    Returns:
        A function that computes the outer product of the weights of the stochastic
        variables.

    """
    arg_names = [f"weight_next_{var}" for var in stochastic_variables]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Array) -> Array:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    return productmap(_outer, variables=tuple(arg_names))


def _get_U_and_F(model: InternalModel) -> Callable[..., tuple[Scalar, Scalar]]:
    """Get the instantaneous utility and feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Args:
        model: The internal model object.

    Returns:
        The instantaneous utility and feasibility function.

    """
    functions = {"feasibility": _get_feasibility(model), **model.functions}
    return concatenate_functions(
        functions=functions,
        targets=["utility", "feasibility"],
        enforce_signature=False,
    )


def _get_feasibility(model: InternalModel) -> InternalUserFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        model: The internal model object.

    Returns:
        The combined constraint function (feasibility).

    """
    constraints = model.function_info.query("is_constraint").index.tolist()

    if constraints:
        combined_constraint = concatenate_functions(
            functions=model.functions,
            targets=constraints,
            aggregator=jnp.logical_and,
        )
    else:

        def combined_constraint(**kwargs: Scalar) -> bool:  # noqa: ARG001
            """Dummy feasibility function that always returns True."""
            return True

    return combined_constraint
