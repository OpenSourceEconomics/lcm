import functools
import inspect
from copy import deepcopy

import jax.numpy as jnp
import pandas as pd
from dags import get_ancestors
from dags.signature import with_signature

import lcm.grids as grids_module
from lcm.create_params import create_params
from lcm.function_evaluator import get_label_translator
from lcm.functools import all_as_args, all_as_kwargs
from lcm.interfaces import GridSpec, Model


def process_model(user_model):
    """Process the user model.

    This entails the following steps:

    - Set defaults where needed
    - Generate derived information
    - Check that the model specification is valid.

    Args:
        user_model (dict): The model as provided by the user.

    Returns:
        lcm.interfaces.Model: The processed model.

    """
    _function_info = _get_function_info(user_model)
    _variable_info = _get_variable_info(
        user_model,
        function_info=_function_info,
    )
    _gridspecs = _get_gridspecs(user_model, variable_info=_variable_info)
    _grids = _get_grids(gridspecs=_gridspecs, variable_info=_variable_info)

    _params = create_params(user_model, variable_info=_variable_info, grids=_grids)

    _functions = _get_functions(
        user_model,
        function_info=_function_info,
        variable_info=_variable_info,
        params=_params,
        grids=_grids,
    )
    return Model(
        grids=_grids,
        gridspecs=_gridspecs,
        variable_info=_variable_info,
        functions=_functions,
        function_info=_function_info,
        params=_params,
        shocks=user_model.get("shocks", {}),
        n_periods=user_model["n_periods"],
    )


def _get_variable_info(user_model, function_info):
    """Derive information about all variables in the model.

    Args:
        user_model (dict): The model as provided by the user.
        function_info (pandas.DataFrame): A table with information about all
            functions in the model. The index contains the name of a function. The
            columns are booleans that are True if the function has the corresponding
            property. The columns are: is_filter, is_constraint, is_next.

    Returns:
        pandas.DataFrame: A table with information about all variables in the model.
            The index contains the name of a model variable. The columns are booleans
            that are True if the variable has the corresponding property. The columns
            are: is_state, is_choice, is_continuous, is_discrete, is_sparse, is_dense.

    """
    _variables = {
        **user_model["states"],
        **user_model["choices"],
    }

    info = pd.DataFrame(index=list(_variables))

    info["is_state"] = info.index.isin(user_model["states"])
    info["is_choice"] = ~info["is_state"]

    info["is_discrete"] = ["options" in spec for spec in _variables.values()]
    info["is_continuous"] = ~info["is_discrete"]

    info["is_stochastic"] = [
        (
            var in user_model["states"]
            and function_info.loc[f"next_{var}", "is_stochastic_next"]
        )
        for var in _variables
    ]

    _auxiliary_variables = _get_auxiliary_variables(
        state_variables=info.query("is_state").index.tolist(),
        function_info=function_info,
        user_functions=user_model["functions"],
    )
    info["is_auxiliary"] = [var in _auxiliary_variables for var in _variables]

    _filtered_variables = set()
    _filter_names = function_info.query("is_filter").index.tolist()

    for name in _filter_names:
        _filtered_variables = _filtered_variables.union(
            get_ancestors(user_model["functions"], name),
        )

    info["is_sparse"] = [var in _filtered_variables for var in _variables]
    info["is_dense"] = ~info["is_sparse"]

    order = info.query("is_sparse & is_state").index.tolist()
    order += info.query("is_sparse & is_choice").index.tolist()
    order += info.query("is_dense & is_discrete & is_state").index.tolist()
    order += info.query("is_dense & is_discrete & is_choice").index.tolist()
    order += info.query("is_continuous & is_state").index.tolist()
    order += info.query("is_continuous & is_choice").index.tolist()

    if set(order) != set(info.index):
        raise ValueError("Order and index do not match.")

    return info.loc[order]


def _get_auxiliary_variables(state_variables, function_info, user_functions):
    """Get state variables that only occur in next functions.

    Args:
        state_variables (list): List of state variable names.
        function_info (pandas.DataFrame): A table with information about all
            functions in the model. The index contains the name of a function. The
            columns are booleans that are True if the function has the corresponding
            property. The columns are: is_filter, is_constraint, is_next.
        user_functions (dict): Dictionary that maps names of functions to functions.

    Returns:
        list: List of state variable names that are only used in next functions.

    """
    non_next_functions = function_info.query("~is_next").index.tolist()
    user_functions = {name: user_functions[name] for name in non_next_functions}
    ancestors = get_ancestors(
        user_functions,
        targets=list(user_functions),
        include_targets=True,
    )
    return list(set(state_variables).difference(set(ancestors)))


def _get_gridspecs(user_model, variable_info):
    """Create a dictionary of grid specifications for each variable in the model.

    Args:
        user_model (dict): The model as provided by the user.
        variable_info (pandas.DataFrame): A table with information about all
            variables in the model. The index contains the name of a model variable.
            The columns are booleans that are True if the variable has the
            corresponding property. The columns are: is_state, is_choice, is_continuous,
            is_discrete, is_sparse, is_dense.

    Returns:
        dict: Dictionary containing all variables of the model. The keys are
            the names of the variables. The values describe which values the variable
            can take. For discrete variables these are the options. For continuous
            variables this is information about how to build the grids.

    """
    raw = {**user_model["states"], **user_model["choices"]}

    variables = {}
    for name, spec in raw.items():
        if "options" in spec:
            variables[name] = spec["options"]
        else:
            variables[name] = GridSpec(
                kind=spec["grid_type"],
                specs={k: v for k, v in spec.items() if k != "grid_type"},
            )

    order = variable_info.index.tolist()
    return {k: variables[k] for k in order}


def _get_grids(gridspecs, variable_info):
    """Create a dictionary of grids for each variable in the model.

    Args:
        gridspecs (dict): Dictionary containing all variables of the model. The keys
            are the names of the variables. The values describe which values the
            variable can take. For discrete variables these are the options. For
            continuous variables this is information about how to build the grids.
        variable_info (pandas.DataFrame): A table with information about all
            variables in the model. The index contains the name of a model variable.
            The columns are booleans that are True if the variable has the
            corresponding property. The columns are: is_state, is_choice, is_continuous,
            is_discrete, is_sparse, is_dense.

    Returns:
        dict: Dictionary containing all variables of the model. The keys are
            the names of the variables. The values are the grids.

    """
    grids = {}
    for name, grid_info in gridspecs.items():
        if variable_info.loc[name, "is_discrete"]:
            grids[name] = jnp.array(grid_info)
        else:
            func = getattr(grids_module, grid_info.kind)
            grids[name] = func(**grid_info.specs)

    order = variable_info.index.tolist()
    return {k: grids[k] for k in order}


def _get_function_info(user_model):
    """Derive information about all functions in the model.

    Args:
        user_model (dict): The model as provided by the user.

    Returns:
        pandas.DataFrame: A table with information about all functions in the model.
            The index contains the name of a model function. The columns are booleans
            that are True if the function has the corresponding property. The columns
            are: is_next, is_filter, is_constraint.

    """
    info = pd.DataFrame(index=list(user_model["functions"]))
    info["is_stochastic_next"] = [
        hasattr(func, "_stochastic_info") for func in user_model["functions"].values()
    ]
    info["is_filter"] = info.index.str.endswith("_filter")
    info["is_constraint"] = info.index.str.endswith("_constraint")
    info["is_next"] = (
        info.index.str.startswith("next_") & ~info["is_constraint"] & ~info["is_filter"]
    )

    return info


def _get_functions(user_model, function_info, variable_info, grids, params):
    """Process the user provided model functions.

    Args:
        user_model (dict): The model as provided by the user.
        function_info (pd.DataFrame): A table with information about model functions.
        variable_info (pd.DataFrame): A table with information about model variables.
        grids (dict): Dictionary containing all variables of the model. The keys are
            the names of the variables. The values are the grids.
        params (dict): The parameters of the model.

    Returns:
        dict: Dictionary containing all functions of the model. The keys are
            the names of the functions. The values are the processed functions.
            The main difference between processed and unprocessed functions is that
            processed functions take `params` as argument unless they are filter
            functions.

    """
    raw_functions = deepcopy(user_model["functions"])

    for var in user_model["states"]:
        if variable_info.loc[var, "is_stochastic"]:
            raw_functions[f"next_{var}"] = _get_stochastic_next_function(
                raw_func=raw_functions[f"next_{var}"],
                grid=grids[var],
            )

            raw_functions[f"weight_next_{var}"] = _get_stochastic_weight_function(
                raw_func=raw_functions[f"next_{var}"],
                name=var,
                variable_info=variable_info,
                grids=grids,
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
            # params[name] contains the dictionary of parameters for the function
            depends_on_params = bool(params[name])

            if is_filter_function:
                if params.get(name, {}):
                    raise ValueError("filters cannot depend on model parameters.")
                processed_func = func

            elif depends_on_params:
                processed_func = _get_extracting_function(
                    func=func,
                    params=params,
                    name=name,
                )

            else:
                processed_func = _get_function_with_dummy_params(func=func)

        functions[name] = processed_func

    return functions


def _get_extracting_function(func, params, name):
    old_signature = list(inspect.signature(func).parameters)
    new_kwargs = [p for p in old_signature if p not in params[name]] + ["params"]

    @with_signature(args=new_kwargs)
    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=new_kwargs)
        _kwargs = {k: v for k, v in kwargs.items() if k in new_kwargs and k != "params"}
        return func(**_kwargs, **kwargs["params"][name])

    return processed_func


def _get_function_with_dummy_params(func):
    old_signature = list(inspect.signature(func).parameters)

    new_kwargs = [*old_signature, "params"]

    @with_signature(args=new_kwargs)
    @functools.wraps(func)
    def processed_func(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=new_kwargs)
        _kwargs = {k: v for k, v in kwargs.items() if k != "params"}
        return func(**_kwargs)

    return processed_func


def _get_stochastic_next_function(raw_func, grid):
    @functools.wraps(raw_func)
    def next_func(*args, **kwargs):  # noqa: ARG001
        return grid

    return next_func


def _get_stochastic_weight_function(raw_func, name, variable_info, grids):
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
        raw_func (callable): The raw next function of the stochastic variable.
        name (str): The name of the stochastic variable.
        variable_info (pd.DataFrame): A table with information about model variables.
        grids (dict): Dictionary containing all variables of the model. The keys are
            the names of the variables. The values are the grids.

    Returns:
        callable: A function that returns the transition weights of the stochastic
            variable.

    """
    function_parameters = list(inspect.signature(raw_func).parameters)

    # Assert that stochastic next function only depends on discrete variables or period
    for arg in function_parameters:
        if arg != "_period" and not variable_info.loc[arg, "is_discrete"]:
            raise ValueError(
                f"Stochastic variables can only depend on discrete variables and "
                f"'_period', but {name} depends on {arg}.",
            )

    label_translators = {
        var: get_label_translator(labels=grids[var], in_name=var)
        for var in function_parameters
        if var != "_period"
    }
    if "_period" in function_parameters:
        label_translators["_period"] = lambda _period: _period

    new_kwargs = [*function_parameters, "params"]

    @with_signature(args=new_kwargs)
    def weight_func(*args, **kwargs):
        args = all_as_args(args, kwargs, arg_names=new_kwargs)
        params = args[-1]  # by definition of new_kargs, params is the last argument

        indices = [
            label_translators[arg_name](arg)
            for arg_name, arg in zip(function_parameters, args, strict=False)
        ]

        return params["shocks"][name][*indices]

    return weight_func
