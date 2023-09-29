"""Create a parameters for a model specification."""
import inspect

import numpy as np


def create_params(user_model, variable_info, grids):
    """Get parameters from a model specification.

    Args:
        user_model (dict): A model specification. Has keys
            - "functions": A dictionary of functions used in the model.
            - "choices": A dictionary of choice variables.
            - "states": A dictionary of state variables.
            - "n_periods": Number of periods in the model (int).
            - "shocks": A dictionary of shock variables (optional).
        variable_info (pd.DataFrame): A dataframe with information about the variables.
        grids (dict): A dictionary of grids.

    Returns:
        dict: A dictionary of model parameters.

    """
    params = {
        **_create_standard_params(),
        **_create_function_params(user_model),
    }

    if variable_info["is_stochastic"].any():
        params["shocks"] = _create_shock_params(
            user_model,
            variable_info=variable_info,
            grids=grids,
        )

    return params


def _create_function_params(model):
    """Get function parameters from a model specification.

    Args:
        model (dict): A model specification. Has keys
            - "functions": A dictionary of functions used in the model.
            - "choices": A dictionary of choice variables.
            - "states": A dictionary of state variables.
            - "n_periods": Number of periods in the model (int).
            - "shocks": A dictionary of shock variables (optional).

    Returns:
        dict: A dictionary of function parameters.

    """
    variables = {
        *model["functions"],
        *model["choices"],
        *model["states"],
    }

    if "shocks" in model:
        variables = variables | set(model["shocks"])
    out = {}
    for name, func in model["functions"].items():
        arguments = set(inspect.signature(func).parameters)
        params = sorted(arguments.difference(variables))
        out[name] = {p: np.nan for p in params}
    return out


def _create_shock_params(model, variable_info, grids):
    """Infer parameters from shocks.

    Args:
        model (dict): A model specification. Has keys
            - "functions": A dictionary of functions used in the model.
            - "choices": A dictionary of choice variables.
            - "states": A dictionary of state variables.
            - "n_periods": Number of periods in the model (int).
            - "shocks": A dictionary of shock variables (optional).
        variable_info (pd.DataFrame): A dataframe with information about the variables.
        grids (dict): A dictionary of grids.

    Returns:
        dict: A dictionary of parameters.

    """
    stochastic_variables = variable_info.query("is_stochastic").index.tolist()

    for var in stochastic_variables:
        if (
            not variable_info.loc[var, "is_state"]
            or not variable_info.loc[var, "is_discrete"]
        ):
            raise ValueError(
                f"Shock {var} is stochastic but not a discrete state variable.",
            )

    params = {}
    for var in stochastic_variables:
        # read signature of next function corresponding to stochastic variable
        dependencies = list(
            inspect.signature(model["functions"][f"next_{var}"]).parameters,
        )

        _check_variables_are_all_discrete_states(
            variables=dependencies,
            variable_info=variable_info,
            msg_suffix=(
                f"The function next_{var} can only depend on discrete state variables."
            ),
        )

        # get dimensions of variables that influence the stochastic variable
        dimensions = [len(grids[dep]) for dep in dependencies]
        # add dimension of stochastic variable to first axis
        dimensions = (len(grids[var]), *dimensions)

        params[var] = np.full(dimensions, np.nan)

    return params


def _create_standard_params():
    return {"beta": np.nan}


def _check_variables_are_all_discrete_states(variables, variable_info, msg_suffix):
    discrete_state_vars = variable_info.query("is_state and is_discrete").index.tolist()
    for var in variables:
        if var not in discrete_state_vars:
            raise ValueError(
                f"Variable {var} is not a discrete state variable. {msg_suffix}",
            )
