import inspect

import jax.numpy as jnp
from dags import concatenate_functions
from jax import vmap

from lcm.function_evaluator import get_function_evaluator


def get_utility_and_feasibility_function(
    model,
    space_info,
    data_name,
    interpolation_options,
    is_last_period,
):
    """Create a func to calculate utility and feasibility of a state choice combination.

    Utility here is not just the current period reward but also includes the discounted
    expected future reward. In the dynamic programming literature this is often denoted
    as capital U, whereas the period reward is lowercase u.

    See `lcm.solve_brute.solve` for more details.

    Notes:
    ------
    - The is last period currently sets continuation values to zero. Needs to be changed
      to accomodate bequest motives.
    - the fake big_u function for last period takes vf_arr, just to make things run for
      to save us some if conditions in the backwards induction loop. Should
      be moved somewhere else:

    Args:
        model (Model): The model object.
        space_info (SpaceInfo): Namedtuple containing all information needed to
            interpret the precalculated values of a function.
        data_name (str): The name of the argument via which the precalculated values
            will be passed into the resulting function.
        interpolation_options (dict): Dictionary of keyword arguments for interpolation
            via map_coordinates.
        is_last_period (bool): Whether the function is created for the last period.

    Returns:
        dict or callable

    """
    if is_last_period:
        # vf_arr is there just so the final period function takes a vf_arr, without
        # using it.
        def _big_u(utility, vf_arr):  # noqa: ARG001
            return utility

    else:

        def _big_u(utility, continuation_value, params):
            return utility + params["beta"] * continuation_value

    feasibility = get_combined_constraint(model)
    next_functions_names, next_state = get_next_state_function(model)
    next_functions = get_dict_of_next_functions(next_functions_names)

    func_dict = {
        "_big_u": _big_u,
        "feasibility": feasibility,
    }
    func_dict.update(next_functions)

    if not is_last_period:
        function_evaluator = get_function_evaluator(
            space_info=space_info,
            data_name=data_name,
            interpolation_options=interpolation_options,
            return_type="dict",
            input_prefix="next_",
            out_name="continuation_value",
        )["functions"]
        func_dict.update(function_evaluator)

    _big_u = concatenate_functions(
        functions=func_dict,
        targets="_big_u",
    )

    feasibility = concatenate_functions(
        functions=func_dict,
        targets="feasibility",
    )

    if not is_last_period:
        parameters = list(inspect.signature(_big_u).parameters)
        next_state_index = parameters.index("next_state")
        in_axes = [
            None,
        ] * len(parameters)
        in_axes[next_state_index] = 0
        _big_u_vmapped = vmap(_big_u, in_axes=in_axes)
        _big_u_vmapped.__signature__ = inspect.signature(_big_u)

    new_func_dict = {
        "_big_u": _big_u,
        "feasibility": feasibility,
        **model.functions,
    }

    if not is_last_period:
        new_func_dict["next_state"] = next_state

    return concatenate_functions(
        functions=new_func_dict,
        targets=["_big_u", "feasibility"],
    )


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


def get_dict_of_next_functions(next_functions_names):
    return {name: get_next_function(name) for name in next_functions_names}


def get_next_function(next_func_name):
    def next_func(next_state):
        return next_state[next_func_name]

    return next_func


# ======================================================================================
# Next state
# ======================================================================================


def get_next_state_function(model):
    """Combine the next state functions into one function.

    Args:
        model (Model): Model instance.

    Returns:
        function: Combined next state function.

    """
    targets = model.function_info.query("is_next").index.tolist()

    return targets, concatenate_functions(
        functions=model.functions,
        targets=targets,
        return_type="dict",
        enforce_signature=True,
    )
