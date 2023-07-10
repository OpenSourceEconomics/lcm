import jax.numpy as jnp
from dags import concatenate_functions

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

    func_dict = {"__big_u__": _big_u, "feasibility": feasibility, **model.functions}

    if not is_last_period:
        func_dict.update(
            get_function_evaluator(
                space_info=space_info,
                data_name=data_name,
                interpolation_options=interpolation_options,
                return_type="dict",
                input_prefix="next_",
                out_name="continuation_value",
            )["functions"],
        )

    return concatenate_functions(
        functions=func_dict,
        targets=["__big_u__", "feasibility"],
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
