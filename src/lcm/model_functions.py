import inspect

import jax.numpy as jnp
from dags import concatenate_functions
from dags.signature import with_signature

from lcm.dispatchers import productmap
from lcm.function_representation import get_function_representation
from lcm.functools import (
    all_as_args,
    all_as_kwargs,
    get_union_of_arguments,
)
from lcm.interfaces import InternalModel
from lcm.next_state import get_next_state_function


def get_utility_and_feasibility_function(
    model: InternalModel,
    space_info,
    name_of_values_on_grid,
    period,
    is_last_period,
):
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
        relevant_functions = [current_u_and_f]

    else:
        next_state = get_next_state_function(model, target="solve")
        next_weights = get_next_weights_function(model)

        scalar_value_function = get_function_representation(
            space_info=space_info,
            name_of_values_on_grid=name_of_values_on_grid,
            input_prefix="next_",
        )

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

    arg_names = {"vf_arr"} | get_union_of_arguments(relevant_functions) - {"_period"}
    arg_names = [arg for arg in arg_names if "next_" not in arg]  # type: ignore[assignment]

    if is_last_period:

        @with_signature(args=arg_names)
        def u_and_f(*args, **kwargs):
            kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)

            states = {k: v for k, v in kwargs.items() if k in state_variables}
            choices = {k: v for k, v in kwargs.items() if k in choice_variables}

            return current_u_and_f(
                **states,
                **choices,
                _period=period,
                params=kwargs["params"],
            )

    else:

        @with_signature(args=arg_names)
        def u_and_f(*args, **kwargs):
            kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)

            states = {k: v for k, v in kwargs.items() if k in state_variables}
            choices = {k: v for k, v in kwargs.items() if k in choice_variables}

            u, f = current_u_and_f(
                **states,
                **choices,
                _period=period,
                params=kwargs["params"],
            )

            _next_state = next_state(
                **states,
                **choices,
                _period=period,
                params=kwargs["params"],
            )
            weights = next_weights(
                **states,
                **choices,
                _period=period,
                params=kwargs["params"],
            )

            value_function = productmap(
                scalar_value_function,
                variables=[f"next_{var}" for var in stochastic_variables],
            )

            ccvs_at_nodes = value_function(
                **_next_state,
                **{k: v for k, v in kwargs.items() if k in value_function_arguments},
            )

            node_weights = multiply_weights(**weights)

            ccv = (ccvs_at_nodes * node_weights).sum()

            big_u = u + kwargs["params"]["beta"] * ccv
            return big_u, f

    return u_and_f


def get_multiply_weights(stochastic_variables):
    """Get multiply_weights function.

    Args:
        stochastic_variables (list): List of stochastic variables.

    Returns:
        callable

    """
    arg_names = [f"weight_next_{var}" for var in stochastic_variables]

    @with_signature(args=arg_names)
    def _outer(*args, **kwargs):
        args = all_as_args(args, kwargs, arg_names=arg_names)
        return jnp.prod(jnp.array(args))

    return productmap(_outer, variables=arg_names)


def get_combined_constraint(model: InternalModel):
    """Create a function that combines all constraint functions into a single one.

    Args:
        model: The internal model object.

    Returns:
        callable

    """
    targets = model.function_info.query("is_constraint").index.tolist()

    if targets:
        combined_constraint = concatenate_functions(
            functions=model.functions,
            targets=targets,
            aggregator=jnp.logical_and,
        )
    else:

        def combined_constraint():
            return None

    return combined_constraint


def get_current_u_and_f(model: InternalModel):
    functions = {"feasibility": get_combined_constraint(model), **model.functions}

    return concatenate_functions(
        functions=functions,
        targets=["utility", "feasibility"],
        enforce_signature=False,
    )


def get_next_weights_function(model: InternalModel):
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
