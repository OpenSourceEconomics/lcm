"""Create a parameters for a model specification."""

import inspect
from typing import Any

import numpy as np
import pandas as pd
from jax import Array
from jax.typing import ArrayLike


def create_params_template(
    model_spec: dict[str, Any],
    variable_info: pd.DataFrame,
    grids: dict[str, Array],
    default_params: dict[str, float] | None = None,
) -> dict[str, ArrayLike]:
    """Create parameter template from a model specification.

    Args:
        model_spec (dict): A model specification provided by the user. Has keys
            - "functions": A dictionary of functions used in the model.
            - "choices": A dictionary of choice variables.
            - "states": A dictionary of state variables.
            - "n_periods": Number of periods in the model (int).
            - "shocks": A dictionary of shock variables (optional).
        variable_info (pd.DataFrame): A dataframe with information about the variables.
        grids (dict): A dictionary of grids retrieved from model_spec.
        default_params (dict): A dictionary of default parameters. Default is None. If
            None, the default {"beta": np.nan} is used. For other lifetime reward
            objectives, additional parameters may be required, for example {"beta":
            np.nan, "delta": np.nan} for beta-delta discounting.

    Returns:
        dict: A dictionary of model parameters.

    """
    if default_params is None:
        # The default lifetime reward objective in LCM is expected discounted utility.
        # For this objective the only additional parameter is the discounting rate beta.
        default_params = {"beta": np.nan}

    params_template = default_params | _create_function_params(model_spec)

    if variable_info["is_stochastic"].any():
        params_template["shocks"] = _create_stochastic_transition_params(
            model_spec=model_spec,
            variable_info=variable_info,
            grids=grids,
        )

    return params_template


def _create_function_params(model_spec: dict[str, Any]) -> dict[str, float]:
    """Get function parameters from a model specification.

    Note: The function argument '_period' is handled separately. It is not treated as a
    parameter, but represents the period of the model.

    Args:
        model_spec (dict): A model specification provided by the user. Has keys
            - "functions": A dictionary of functions used in the model.
            - "choices": A dictionary of choice variables.
            - "states": A dictionary of state variables.
            - "n_periods": Number of periods in the model (int).
            - "shocks": A dictionary of shock variables (optional).

    Returns:
        dict: A dictionary of parameters required in the model functions, initialized
            with np.nan.

    """
    # Collect all model variables, that includes choices, states, the period, and
    # auxiliary variables (model function names).
    variables = {
        *model_spec["functions"],
        *model_spec["choices"],
        *model_spec["states"],
        "_period",
    }

    if "shocks" in model_spec:
        variables = variables | set(model_spec["shocks"])

    function_params = {}
    # For each model function, capture the arguments of the function that are not in the
    # set of model variables, and initialize them.
    for name, func in model_spec["functions"].items():
        arguments = set(inspect.signature(func).parameters)
        params = sorted(arguments.difference(variables))
        function_params[name] = {p: np.nan for p in params}

    return function_params


def _create_stochastic_transition_params(
    model_spec: dict[str, Any],
    variable_info: pd.DataFrame,
    grids: dict[str, Array],
) -> dict[str, ArrayLike]:
    """Create parameters for stochastic transitions.

    Args:
        model_spec (dict): A model specification provided by the user. Has keys
            - "functions": A dictionary of functions used in the model.
            - "choices": A dictionary of choice variables.
            - "states": A dictionary of state variables.
            - "n_periods": Number of periods in the model (int).
            - "shocks": A dictionary of shock variables (optional).
        variable_info (pd.DataFrame): A dataframe with information about the variables.
        grids (dict): A dictionary of grids retrieved from model_spec

    Returns:
        dict: A dictionary of parameters required for stochastic transitions,
            initialized with np.nan matrices of the correct dimensions.

    """
    stochastic_variables = variable_info.query("is_stochastic").index.tolist()

    # ----------------------------------------------------------------------------------
    # Assert that all stochastic variables are discrete state variables
    # ----------------------------------------------------------------------------------
    discrete_state_vars = set(variable_info.query("is_state & is_discrete").index)

    invalid = set(stochastic_variables) - discrete_state_vars

    if invalid:
        raise ValueError(
            f"The following variables are stochastic, but are not discrete state "
            f"variables: {invalid}. This is currently not supported.",
        )

    # ----------------------------------------------------------------------------------
    # Create template matrices for stochastic transitions
    # ----------------------------------------------------------------------------------

    # Stochastic transition functions can only depend on discrete vars or '_period'.
    valid_vars = set(variable_info.query("is_discrete").index) | {"_period"}

    stochastic_transition_params = {}
    invalid_dependencies = {}

    for var in stochastic_variables:

        # Retrieve corresponding next function and its arguments
        next_var = model_spec["functions"][f"next_{var}"]
        dependencies = list(inspect.signature(next_var).parameters)

        invalid = set(dependencies) - valid_vars
        # If there are invalid dependencies, store them in a dictionary and continue
        # with the next variable to collect as many invalid arguments as possible.
        if invalid:
            invalid_dependencies[var] = invalid
            continue

        # Get the dimensions of variables that influence the stochastic variable
        dimensions_of_deps = [
            len(grids[arg]) if arg != "_period" else model_spec["n_periods"]
            for arg in dependencies
        ]
        # Add the dimension of the stochastic variable itself at the end
        dimensions = (*dimensions_of_deps, len(grids[var]))

        stochastic_transition_params[var] = np.full(dimensions, np.nan)

    # ----------------------------------------------------------------------------------
    # Raise an error if there are invalid arguments
    # ----------------------------------------------------------------------------------
    if invalid_dependencies:
        raise ValueError(
            f"Stochastic transition functions can only depend on discrete variables or "
            "'_period'. The following variables have invalid arguments: "
            f"{invalid_dependencies}.",
        )

    return stochastic_transition_params
