import inspect
from collections.abc import Callable

import jax.numpy as jnp
from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array

from lcm.dispatchers import productmap
from lcm.function_representation import get_value_function_representation
from lcm.functools import (
    all_as_args,
    all_as_kwargs,
    get_union_of_arguments,
)
from lcm.interfaces import InternalModel, StateSpaceInfo
from lcm.next_state import get_next_state_function
from lcm.typing import InternalUserFunction, ParamsDict, Scalar, Target


def get_utility_and_feasibility_function(
    model: InternalModel,
    state_space_info: StateSpaceInfo,
    period: int,
    *,
    is_last_period: bool,
) -> Callable[..., tuple[Array, Array]]:
    # ==================================================================================
    # Gather information on the model variables
    # ==================================================================================
    state_variables = model.variable_info.query("is_state").index.tolist()
    choice_variables = model.variable_info.query("is_choice").index.tolist()
    stochastic_variables = model.variable_info.query("is_stochastic").index.tolist()

    # ==================================================================================
    # Generate dynamic functions
    # ==================================================================================
    current_u_and_f = get_current_u_and_f(model)

    if is_last_period:
        relevant_functions: list[
            Callable[..., Scalar]
            | Callable[..., tuple[Scalar, Scalar]]
            | Callable[..., dict[str, Scalar]]
        ] = [current_u_and_f]

    else:
        next_state = get_next_state_function(model, target=Target.SOLVE)
        next_weights = get_next_weights_function(model)

        scalar_value_function = get_value_function_representation(state_space_info)

        multiply_weights = get_multiply_weights(stochastic_variables)

        relevant_functions = [
            current_u_and_f,
            next_state,
            next_weights,
            scalar_value_function,
        ]

        value_function_arguments = list(
            inspect.signature(scalar_value_function).parameters,
        )

    # ==================================================================================
    # Create the utility and feasability function
    # ==================================================================================

    arg_names_set = {"vf_arr"} | get_union_of_arguments(relevant_functions) - {
        "_period"
    }
    arg_names = [arg for arg in arg_names_set if "next_" not in arg]

    if is_last_period:

        @with_signature(args=arg_names)
        def u_and_f(
            *args: Scalar, params: ParamsDict, **kwargs: Scalar
        ) -> tuple[Scalar, Scalar]:
            kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)

            states = {k: v for k, v in kwargs.items() if k in state_variables}
            choices = {k: v for k, v in kwargs.items() if k in choice_variables}

            return current_u_and_f(
                **states,
                **choices,
                _period=period,
                params=params,
            )

    else:

        @with_signature(args=arg_names)
        def u_and_f(
            *args: Scalar, params: ParamsDict, **kwargs: Scalar
        ) -> tuple[Scalar, Scalar]:
            kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)

            states = {k: v for k, v in kwargs.items() if k in state_variables}
            choices = {k: v for k, v in kwargs.items() if k in choice_variables}

            u, f = current_u_and_f(
                **states,
                **choices,
                _period=period,
                params=params,
            )

            _next_state = next_state(
                **states,
                **choices,
                _period=period,
                params=params,
            )
            weights = next_weights(
                **states,
                **choices,
                _period=period,
                params=params,
            )

            value_function = productmap(
                scalar_value_function,
                variables=tuple(f"next_{var}" for var in stochastic_variables),
            )

            ccvs_at_nodes = value_function(
                **_next_state,
                **{k: v for k, v in kwargs.items() if k in value_function_arguments},
            )

            node_weights = multiply_weights(**weights)

            ccv = (ccvs_at_nodes * node_weights).sum()

            big_u = u + params["beta"] * ccv
            return big_u, f

    return u_and_f


def get_multiply_weights(stochastic_variables: list[str]) -> Callable[..., Array]:
    """Get multiply_weights function.

    Args:
        stochastic_variables (list): List of stochastic variables.

    Returns:
        A function that multiplies the weights of the stochastic variables.

    """
    arg_names = [f"weight_next_{var}" for var in stochastic_variables]

    @with_signature(args=arg_names)
    def _outer(*args: Array, **kwargs: Array) -> Array:
        args = all_as_args(args, kwargs, arg_names=arg_names)
        return jnp.prod(jnp.array(args))

    return productmap(_outer, variables=tuple(arg_names))


def get_combined_constraint(model: InternalModel) -> InternalUserFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        model: The internal model object.

    Returns:
        The combined constraint function.

    """
    targets = model.function_info.query("is_constraint").index.tolist()

    if targets:
        combined_constraint = concatenate_functions(
            functions=model.functions,
            targets=targets,
            aggregator=jnp.logical_and,
        )
    else:

        def combined_constraint() -> None:
            return None

    return combined_constraint


def get_current_u_and_f(model: InternalModel) -> Callable[..., tuple[Scalar, Scalar]]:
    functions = {"feasibility": get_combined_constraint(model), **model.functions}

    return concatenate_functions(
        functions=functions,
        targets=["utility", "feasibility"],
        enforce_signature=False,
    )


def get_next_weights_function(model: InternalModel) -> Callable[..., dict[str, Scalar]]:
    targets = [
        f"weight_{name}"
        for name in model.function_info.query("is_stochastic_next").index.tolist()
    ]

    return concatenate_functions(
        functions=model.functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
    )
