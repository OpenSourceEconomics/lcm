"""Generate functions that compute the next states of the model.

For the solution, we simply concatenate the functions that compute the next states. For
the simulation, we generate functions that simulate the next states of stochastic
variables. We then concatenate these functions with the functions that compute the
deteministic next states.

"""

from collections.abc import Callable

from dags import concatenate_functions
from dags.signature import with_signature
from jax import Array

from lcm.functools import all_as_args
from lcm.interfaces import InternalModel
from lcm.random_choice import random_choice
from lcm.typing import Scalar, Target


def get_next_state_function(
    model: InternalModel, target: Target
) -> Callable[..., dict[str, Scalar]]:
    """Get function that computes the next states of the model.

    Args:
        model: Internal model.
        target: Target of the function.

    Returns:
        Function that computes the next states of the model.

    """
    if target == Target.SOLVE:
        return _get_next_state_function_for_solution(model)

    if target == Target.SIMULATE:
        return _get_next_state_function_for_simulation(model)

    raise ValueError(f"Invalid target: {target}")


# ======================================================================================
# Solution
# ======================================================================================


def _get_next_state_function_for_solution(
    model: InternalModel,
) -> Callable[..., dict[str, Scalar]]:
    """Get function that computes the next states for the solution.

    Args:
        model: Model instance.

    Returns:
        Function that computes the next states. Depends on states and choices of the
        current period, and the model parameters.

    """
    targets = model.function_info.query("is_next").index.tolist()

    return concatenate_functions(
        functions=model.functions,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
    )


# ======================================================================================
# Simulation
# ======================================================================================


def _get_next_state_function_for_simulation(
    model: InternalModel,
) -> Callable[..., dict[str, Scalar]]:
    """Get function that computes the next states for the simulation.

    Args:
        model: Model instance.

    Returns:
        Function that computes the next states. Depends on states and choices of the
        current period, and the model parameters. Additionaly, it depends on:
        - key (dict): Dictionary with PRNG keys. Keys are the names of stochastic next
          functions, e.g. 'next_health'.

    """
    # ==================================================================================
    # Get targets
    # ==================================================================================
    targets = model.function_info.query("is_next").index.tolist()

    stochastic_targets = model.function_info.query(
        "is_next & is_stochastic_next",
    ).index

    # ==================================================================================
    # Handle stochastic next states functions
    # ----------------------------------------------------------------------------------
    # We generate stochastic next states functions that simulate the next state given
    # a PRNG key and the weights of the stochastic variable. The corresponding weights
    # are computed using the stochastic weight functions, which we add the to functions
    # dict. `dags.concatenate_functions` then generates a function that computes the
    # weights and simulates the next state.
    # ==================================================================================
    stochastic_next = {
        name: _get_stochastic_next_func(name, grids=model.grids)
        for name in stochastic_targets
    }

    stochastic_weights_names = [
        f"weight_{name}"
        for name in model.function_info.query("is_stochastic_next").index.tolist()
    ]

    stochastic_weights = {
        name: model.functions[name] for name in stochastic_weights_names
    }

    # ==================================================================================
    # Overwrite model.functions with generated stochastic next states functions
    # ==================================================================================
    functions_dict = model.functions | stochastic_next | stochastic_weights

    return concatenate_functions(
        functions=functions_dict,
        targets=targets,
        return_type="dict",
        enforce_signature=False,
    )


def _get_stochastic_next_func(
    name: str, grids: dict[str, Array]
) -> Callable[[Array, Array], Array]:
    """Get function that simulates the next state of a stochastic variable.

    Args:
        name: Name of the stochastic variable.
        grids: Dict with grids.

    Returns:
        A function that simulates the next state of the stochastic variable. Depends on
        variables:
        - key (dict): Dictionary with PRNG keys. Keys are the names of stochastic next
          functions, e.g. 'next_health'.
        - weight_{name} (jax.numpy.array): 2d array of weights. The first dimension
          corresponds to the number of simulation units. The second dimension
          corresponds to the number of grid points (labels).

    """
    arg_names = ["keys", f"weight_{name}"]
    labels = grids[name.removeprefix("next_")]

    @with_signature(args=arg_names)
    def _next_stochastic_state(*args: Array, **kwargs: Array) -> Array:
        keys, weights = all_as_args(args, kwargs, arg_names=arg_names)
        return random_choice(
            key=keys[name],
            probs=weights,
            labels=labels,
        )

    return _next_stochastic_state
