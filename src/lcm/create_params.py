import inspect

import numpy as np

from lcm import distributions


def create_params(model):
    params = {
        **_create_standard_params(),
        **_create_function_params(model),
    }

    if "shocks" in model:
        params = {**params, **_create_shock_params(model["shocks"])}

    return params


def _create_standard_params():
    return {"beta": np.nan}


def _create_function_params(model):
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
    out = {}
    for name, dist in shocks.items():
        out[name] = getattr(distributions, f"get_{dist}_params")()

    return out
