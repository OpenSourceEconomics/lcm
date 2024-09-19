import functools
import inspect
from collections.abc import Callable
from copy import deepcopy

import pandas as pd
from dags.signature import with_signature
from jax import Array

from lcm.functools import all_as_args, all_as_kwargs
from lcm.input_processing.create_params_template import create_params_template
from lcm.input_processing.discrete_state_conversion import (
    convert_discrete_codes_to_indices,
)
from lcm.input_processing.util import (
    get_function_info,
    get_grids,
    get_gridspecs,
    get_variable_info,
)
from lcm.interfaces import InternalModel
from lcm.typing import ParamsDict, ShockType
from lcm.user_model import Model


def process_model(model: Model) -> InternalModel:
    """Process the user model.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the model specification is valid.

    Args:
        model: The model as provided by the user.

    Returns:
        The processed model.

    """
    tmp_model, converter = convert_discrete_codes_to_indices(model)

    params = create_params_template(tmp_model)

    return InternalModel(
        grids=get_grids(tmp_model),
        gridspecs=get_gridspecs(tmp_model),
        variable_info=get_variable_info(tmp_model),
        functions=_get_internal_functions(tmp_model, params=params),
        function_info=get_function_info(tmp_model),
        params=params,
        converter=converter,
        # currently no additive utility shocks are supported
        random_utility_shocks=ShockType.NONE,
        n_periods=tmp_model.n_periods,
    )


def _get_internal_functions(
    model: Model,
    params: ParamsDict,
) -> dict[str, Callable]:
    """Process the user provided model functions.

    Args:
        model: The model as provided by the user.
        params: The parameters of the model.

    Returns:
        dict: Dictionary containing all functions of the model. The keys are
            the names of the functions. The values are the processed functions.
            The main difference between processed and unprocessed functions is that
            processed functions take `params` as argument unless they are filter
            functions.

    """
    variable_info = get_variable_info(model)
    grids = get_grids(model)
    function_info = get_function_info(model)

    raw_functions = deepcopy(model.functions)

    for var in model.states:
        if variable_info.loc[var, "is_stochastic"]:
            raw_functions[f"next_{var}"] = _get_stochastic_next_function(
                raw_func=raw_functions[f"next_{var}"],
                grid=grids[var],
            )

            raw_functions[f"weight_next_{var}"] = _get_stochastic_weight_function(
                raw_func=raw_functions[f"next_{var}"],
                name=var,
                variable_info=variable_info,
            )

    # ==================================================================================
    # Add 'params' argument to functions
    # ==================================================================================
    # We wrap the user functions such that they can be called with the 'params' argument
    # instead of the individual parameters. This is done for all functions except for
    # filter functions, because they cannot depend on model parameters; and dynamically
    # generated weighting functions for stochastic next functions, since they are
    # constructed to accept the 'params' argument by default.

    functions = {}
    for name, func in raw_functions.items():
        is_weight_next_function = name.startswith("weight_next_")

        if is_weight_next_function:
            processed_func = func

        else:
            is_filter_function = function_info.loc[name, "is_filter"]
            # params[name] contains the dictionary of parameters for the function, which
            # is empty if the function does not depend on any model parameters.
            depends_on_params = bool(params[name])

            if is_filter_function:
                if params.get(name, False):
                    raise ValueError(
                        f"filters cannot depend on model parameters, but {name} does."
                    )
                processed_func = func

            elif depends_on_params:
                processed_func = _replace_func_parameters_by_params(
                    func=func,
                    params=params,
                    name=name,
                )

            else:
                processed_func = _add_dummy_params_argument(func)

        functions[name] = processed_func

    return functions


def _replace_func_parameters_by_params(
    func: Callable, params: ParamsDict, name: str
) -> Callable:
    old_signature = list(inspect.signature(func).parameters)
    new_kwargs = [
        p
        for p in old_signature
        if p not in params[name]  # type: ignore[operator]
    ] + ["params"]

    @with_signature(args=new_kwargs)
    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=new_kwargs)
        _kwargs = {k: v for k, v in kwargs.items() if k in new_kwargs and k != "params"}
        return func(**_kwargs, **kwargs["params"][name])

    return processed_func


def _add_dummy_params_argument(func: Callable) -> Callable:
    old_signature = list(inspect.signature(func).parameters)

    new_kwargs = [*old_signature, "params"]

    @with_signature(args=new_kwargs)
    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=new_kwargs)
        _kwargs = {k: v for k, v in kwargs.items() if k != "params"}
        return func(**_kwargs)

    return processed_func


def _get_stochastic_next_function(raw_func: Callable, grid: Array):
    @functools.wraps(raw_func)
    def next_func(*args, **kwargs):  # noqa: ARG001
        return grid

    return next_func


def _get_stochastic_weight_function(
    raw_func: Callable, name: str, variable_info: pd.DataFrame
):
    """Get a function that returns the transition weights of a stochastic variable.

    Example:
    Consider a stochastic variable 'health' that takes two values {0, 1}. The transition
    matrix is thus 2x2. We create the weighting function and then select the weights
    that correspond to the case where 'health' is 0.

    >>> from lcm.mark import StochasticInfo
    >>> def next_health(health):
    >>>     pass
    >>> next_health._stochastic_info = StochasticInfo()
    >>> params = {"shocks": {"health": np.arange(4).reshape(2, 2)}}
    >>> weight_func = _get_stochastic_weight_function(
    >>>     raw_func=next_health,
    >>>     name="health"
    >>>     variable_info=variable_info,
    >>>     grids=grids,
    >>> )
    >>> weight_func(health=0, params=params)
    >>> array([0, 1])


    Args:
        raw_func: The raw next function of the stochastic variable.
        name: The name of the stochastic variable.
        variable_info: A table with information about model variables.

    Returns:
        callable: A function that returns the transition weights of the stochastic
            variable.

    """
    function_parameters = list(inspect.signature(raw_func).parameters)

    # Assert that stochastic next function only depends on discrete variables or period
    invalid = {
        arg
        for arg in function_parameters
        if arg != "_period" and not variable_info.loc[arg, "is_discrete"]
    }

    if invalid:
        raise ValueError(
            "Stochastic variables can only depend on discrete variables and '_period', "
            f"but {name} depends on {invalid}.",
        )

    new_kwargs = [*function_parameters, "params"]

    @with_signature(args=new_kwargs)
    def weight_func(*args, **kwargs):
        args = all_as_args(args, kwargs, arg_names=new_kwargs)
        params = args[-1]  # by definition of new_kargs, params is the last argument
        # By assumption, all discrete variables that LCM handles internally are
        # themselves indices (i.e. a range starting at 0 with a step size of 1).
        # Therefore, the arguments themselves are the indices. For the special variable
        # '_period' the same holds.
        indices = args[:-1]
        return params["shocks"][name][*indices]

    return weight_func
