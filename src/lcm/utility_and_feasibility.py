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


def get_utility_and_feasibility_function(
    model: InternalModel,
    next_state_space_info: StateSpaceInfo,
    period: int,
    *,
    is_last_period: bool,
) -> Callable[..., tuple[Array, Array]]:
    """Create the utility and feasibility function for a given period.

    Args:
        model: The internal model object.
        next_state_space_info: The state space information of the next period.
        period: The period to create the utility and feasibility function for.
        is_last_period: Whether the period is the last period.

    Returns:
        A function that computes the expected forward-looking utility and feasibility
        for the given period.

    """
    if is_last_period:
        return get_utility_and_feasibility_function_last_period(model, period=period)
    return get_utility_and_feasibility_function_before_last_period(
        model, next_state_space_info=next_state_space_info, period=period
    )


def get_utility_and_feasibility_function_before_last_period(
    model: InternalModel,
    next_state_space_info: StateSpaceInfo,
    period: int,
) -> Callable[..., tuple[Array, Array]]:
    """Create the utility and feasibility function for a period before the last period.

    Args:
        model: The internal model object.
        next_state_space_info: The state space information of the next period.
        period: The period to create the utility and feasibility function for.

    Returns:
        A function that computes the utility and feasibility for the given period.

    """
    stochastic_variables = model.variable_info.query("is_stochastic").index.tolist()

    # ----------------------------------------------------------------------------------
    # Generate dynamic functions
    # ----------------------------------------------------------------------------------
    # TODO (@timmens): This can be done outside this function, since it  # noqa: TD003
    # does not depend on the period.
    # ----------------------------------------------------------------------------------

    # Functions required to calculate the expected continuation values
    calculate_state_transition = get_next_state_function(model, target=Target.SOLVE)
    calculate_next_weights = get_next_stochastic_weights_function(model)
    calculate_node_weights = _get_node_weights_function(stochastic_variables)
    _scalar_value_function = get_value_function_representation(next_state_space_info)
    value_function = productmap(
        _scalar_value_function,
        variables=tuple(f"next_{var}" for var in stochastic_variables),
    )

    # Function required to calculate todays utility and feasibility
    calculate_todays_u_and_f = _get_current_u_and_f(model)

    # ----------------------------------------------------------------------------------
    # Create the utility and feasibility function
    # ----------------------------------------------------------------------------------

    arg_names = _get_required_arg_names_of_u_and_f(
        [
            calculate_todays_u_and_f,
            calculate_state_transition,
            calculate_next_weights,
        ]
    )

    @with_signature(args=arg_names)
    def utility_and_feasibility(
        params: ParamsDict, vf_arr: Array, **states_and_choices: Scalar
    ) -> tuple[Scalar, Scalar]:
        """Calculate the expected forward-looking utility and feasibility.

        Args:
            params: The parameters.
            vf_arr: The value function array.
            **states_and_choices: Todays states and choices.

        Returns:
            A tuple containing the utility and feasibility for the given period.

        """
        # ------------------------------------------------------------------------------
        # Calculate the expected continuation values
        # ------------------------------------------------------------------------------
        next_states = calculate_state_transition(
            **states_and_choices,
            _period=period,
            params=params,
        )

        weights = calculate_next_weights(
            **states_and_choices,
            _period=period,
            params=params,
        )

        node_weights = calculate_node_weights(**weights)

        # As we productmap'd the value function over the stochastic variables, the
        # resulting continuation values get a new dimension for each stochastic
        # variable.
        continuation_values_at_nodes = value_function(**next_states, vf_arr=vf_arr)

        # We then weight these continuation values with the joint node weights and sum
        # them up to get the expected continuation values.
        expected_continuation_values = (
            continuation_values_at_nodes * node_weights
        ).sum()

        # ------------------------------------------------------------------------------
        # Calculate the expected forward-looking utility.
        # ------------------------------------------------------------------------------
        # This is not the value function yet, as it still depends on the choices.
        # ------------------------------------------------------------------------------
        period_utility, period_feasibility = calculate_todays_u_and_f(
            **states_and_choices,
            _period=period,
            params=params,
        )

        expected_forward_utility = (
            period_utility + params["beta"] * expected_continuation_values
        )

        return expected_forward_utility, period_feasibility

    return utility_and_feasibility


def get_utility_and_feasibility_function_last_period(
    model: InternalModel,
    period: int,
) -> Callable[..., tuple[Array, Array]]:
    """Create the utility and feasibility function for the last period.

    Args:
        model: The internal model object.
        period: The period to create the utility and feasibility function for. This is
            still relevant for the last period, as some functions might depend on the
            actual period value.

    Returns:
        A function that computes the utility and feasibility for the given period.

    """
    calculate_todays_u_and_f = _get_current_u_and_f(model)

    arg_names = _get_required_arg_names_of_u_and_f([calculate_todays_u_and_f])

    @with_signature(args=arg_names)
    def utility_and_feasibility(
        params: ParamsDict, **kwargs: Scalar
    ) -> tuple[Scalar, Scalar]:
        return calculate_todays_u_and_f(
            **kwargs,
            _period=period,
            params=params,
        )

    return utility_and_feasibility


# ======================================================================================
# Helper functions
# ======================================================================================


def _get_required_arg_names_of_u_and_f(
    model_functions: list[Callable[..., Any]],
) -> list[str]:
    """Get the argument names of the utility and feasibility function.

    Args:
        model_functions: The list of functions that are used to calculate the utility
            and feasibility.

    Returns:
        The argument names of the utility and feasibility function.

    """
    dynamic_arg_names = get_union_of_arguments(model_functions) - {"_period"}
    static_arg_names = {"params", "vf_arr"}

    return list(static_arg_names | dynamic_arg_names)


def _get_node_weights_function(stochastic_variables: list[str]) -> Callable[..., Array]:
    """Get joint weights function.

    This function takes the weights of the individual stochastic variables and
    multiplies them together to get the joint weights on the product space of the
    stochastic variables.

    Args:
        stochastic_variables: List of stochastic variables.

    Returns:
        A function that multiplies the weights of the stochastic variables.

    """
    arg_names = [f"weight_next_{var}" for var in stochastic_variables]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Array) -> Array:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    return productmap(_outer, variables=tuple(arg_names))


def _get_current_u_and_f(model: InternalModel) -> Callable[..., tuple[Scalar, Scalar]]:
    """Get the current utility and feasibility function.

    Args:
        model: The internal model object.

    Returns:
        The current utility and feasibility function.

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
    targets = model.function_info.query("is_constraint").index.tolist()

    if targets:
        combined_constraint = concatenate_functions(
            functions=model.functions,
            targets=targets,
            aggregator=jnp.logical_and,
        )
    else:

        def combined_constraint(**kwargs: Scalar) -> bool:  # noqa: ARG001
            """Dummy feasibility function that always returns True."""
            return True

    return combined_constraint
