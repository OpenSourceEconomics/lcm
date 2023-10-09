import inspect

import jax.numpy as jnp
from dags import concatenate_functions
from dags.signature import with_signature

from lcm.dispatchers import productmap
from lcm.function_evaluator import get_function_evaluator
from lcm.functools import (
    all_as_args,
    all_as_kwargs,
    get_union_of_arguments,
)
from lcm.next_state import get_next_state_function


def get_utility_and_feasibility_function(
    model,
    space_info,
    data_name,
    interpolation_options,
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

        scalar_value_function = get_function_evaluator(
            space_info=space_info,
            data_name=data_name,
            interpolation_options=interpolation_options,
            input_prefix="next_",
            out_name="continuation_value",
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

    arg_names = {"vf_arr"} | get_union_of_arguments(relevant_functions)
    arg_names = [arg for arg in arg_names if "next_" not in arg]

    if is_last_period:

        @with_signature(args=arg_names)
        def u_and_f(*args, **kwargs):
            kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)

            states = {k: v for k, v in kwargs.items() if k in state_variables}
            choices = {k: v for k, v in kwargs.items() if k in choice_variables}

            return current_u_and_f(**states, **choices, params=kwargs["params"])

    else:

        @with_signature(args=arg_names)
        def u_and_f(*args, **kwargs):
            kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)

            states = {k: v for k, v in kwargs.items() if k in state_variables}
            choices = {k: v for k, v in kwargs.items() if k in choice_variables}

            u, f = current_u_and_f(**states, **choices, params=kwargs["params"])

            _next_state = next_state(**states, **choices, params=kwargs["params"])
            weights = next_weights(**states, params=kwargs["params"])

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


def get_combined_constraint(model):
    """Create a function that combines all constraint functions into a single one.

    Args:
        model (Model): The model object.

    Returns:
        callable

    """
    targets = model.function_info.query("is_constraint").index.tolist()

    return concatenate_functions(
        functions=model.functions,
        targets=targets,
        aggregator=jnp.logical_and,
    )


def get_current_u_and_f(model):
    functions = {"feasibility": get_combined_constraint(model), **model.functions}

    return concatenate_functions(
        functions=functions,
        targets=["utility", "feasibility"],
    )


def get_next_weights_function(model):
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
