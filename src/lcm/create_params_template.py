"""Create a parameter template for a model specification."""

import inspect

import jax.numpy as jnp
import pandas as pd
from jax import Array

from lcm.model import Model
from lcm.typing import Params, ScalarUserInput


def create_params_template(
    user_model: Model,
    variable_info: pd.DataFrame,
    grids: dict[str, Array],
    default_params: dict[str, ScalarUserInput] | None = None,
) -> Params:
    """Create parameter template from a model specification.

    Args:
        user_model: The model as provided by the user.
        variable_info: A dataframe with information about the variables.
        grids: A dictionary of grids consistent with user_model.
        default_params: A dictionary of default parameters. Default is None. If None,
            the default {"beta": np.nan} is used. For other lifetime reward objectives,
            additional parameters may be required, for example {"beta": np.nan, "delta":
            np.nan} for beta-delta discounting.

    Returns:
        dict: A nested dictionary of model parameters.

    """
    if default_params is None:
        # The default lifetime reward objective in LCM is expected discounted utility.
        # For this objective the only additional parameter is the discounting rate beta.
        default_params = {"beta": jnp.nan}

    if variable_info["is_stochastic"].any():
        stochastic_transitions = _create_stochastic_transition_params(
            user_model=user_model,
            variable_info=variable_info,
            grids=grids,
        )
        stochastic_transition_params = {"shocks": stochastic_transitions}
    else:
        stochastic_transition_params = {}

    function_params = _create_function_params(user_model)

    return default_params | function_params | stochastic_transition_params


def _create_function_params(user_model: Model) -> Params:
    """Get function parameters from a model specification.

    Explanation: We consider the arguments of all model functions, from which we exclude
    all variables that are states, choices or the period argument. Everything else is
    considered a parameter of the respective model function that is provided by the
    user.

    Args:
        user_model: The model as provided by the user.

    Returns:
        dict: A dictionary for each model function, containing a parameters required in
            the model functions, initialized with jnp.nan.

    """
    # Collect all model variables, that includes choices, states, the period, and
    # auxiliary variables (model function names).
    variables = {
        *user_model.functions,
        *user_model.choices,
        *user_model.states,
        "_period",
    }

    if hasattr(user_model, "shocks"):
        variables = variables | set(user_model.shocks)

    function_params: Params = {}
    # For each model function, capture the arguments of the function that are not in the
    # set of model variables, and initialize them.
    for name, func in user_model.functions.items():
        arguments = set(inspect.signature(func).parameters)
        params = sorted(arguments.difference(variables))
        function_params[name] = {p: jnp.nan for p in params}

    return function_params


def _create_stochastic_transition_params(
    user_model: Model,
    variable_info: pd.DataFrame,
    grids: dict[str, Array],
) -> dict[str, Array]:
    """Create parameters for stochastic transitions.

    Args:
        user_model: The model as provided by the user.
        variable_info: A dataframe with information about the variables.
        grids: A dictionary of grids consistent with user_model.

    Returns:
        dict: A dictionary of parameters required for stochastic transitions,
            initialized with jnp.nan matrices of the correct dimensions.

    """
    stochastic_variables = variable_info.query("is_stochastic").index.tolist()

    # Assert that all stochastic variables are discrete state variables
    # ==================================================================================
    discrete_state_vars = set(variable_info.query("is_state & is_discrete").index)

    if invalid := set(stochastic_variables) - discrete_state_vars:
        raise ValueError(
            f"The following variables are stochastic, but are not discrete state "
            f"variables: {invalid}. This is currently not supported.",
        )

    # Create template matrices for stochastic transitions
    # ==================================================================================

    # Stochastic transition functions can only depend on discrete vars or '_period'.
    valid_vars = set(variable_info.query("is_discrete").index) | {"_period"}

    stochastic_transition_params = {}
    invalid_dependencies = {}

    for var in stochastic_variables:
        # Retrieve corresponding next function and its arguments
        next_var = user_model.functions[f"next_{var}"]
        dependencies = list(inspect.signature(next_var).parameters)

        # If there are invalid dependencies, store them in a dictionary and continue
        # with the next variable to collect as many invalid arguments as possible.
        if invalid := set(dependencies) - valid_vars:
            invalid_dependencies[var] = invalid
        else:
            # Get the dimensions of variables that influence the stochastic variable
            dimensions_of_deps = [
                len(grids[arg]) if arg != "_period" else user_model.n_periods
                for arg in dependencies
            ]
            # Add the dimension of the stochastic variable itself at the end
            dimensions = (*dimensions_of_deps, len(grids[var]))

            stochastic_transition_params[var] = jnp.full(dimensions, jnp.nan)

    # Raise an error if there are invalid arguments
    # ==================================================================================
    if invalid_dependencies:
        raise ValueError(
            f"Stochastic transition functions can only depend on discrete variables or "
            "'_period'. The following variables have invalid arguments: "
            f"{invalid_dependencies}.",
        )

    return stochastic_transition_params
