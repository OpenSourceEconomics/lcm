"""Create a parameters for a model specification."""
import inspect

import numpy as np

from lcm import distributions


def create_params(model):
    """Get parameters from a model specification.

    Args:
        model (dict): A model specification. Has keys
            - "functions": A dictionary of functions used in the model.
            - "choices": A dictionary of choice variables.
            - "states": A dictionary of state variables.
            - "n_periods": Number of periods in the model (int).
            - "shocks": A dictionary of shock variables (optional).

    Returns:
        dict: A dictionary of model parameters.

    """
    params = {
        **_create_standard_params(),
        **_create_function_params(model),
    }

    if "shocks" in model:
        params = {**params, **_create_shock_params(model["shocks"])}

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


def _create_shock_params(shocks):
    """Infer parameters from shocks.

    Args:
        shocks (dict): A dictionary of shock variables.

    Returns:
        dict: A dictionary of parameters.

    """
    out = {}
    for name, dist in shocks.items():
        out[name] = getattr(distributions, f"get_{dist}_params")()

    return out


def _create_standard_params():
    return {"beta": np.nan}
