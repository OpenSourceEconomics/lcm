"""Create a state space for a given model."""
import inspect

import numpy as np
from lcm import grids as grids_module


def create_state_choice_space(model):
    """Create a state choice space for the model.

    A state_choice_space is a compressed representation of all feasible states and the
    feasible choices within that state. We currently use the following compressions:

    - We distinguish between simple and complex variables. Simple state or choice
      variables are those whose set of feasible values does not depend on any other
      state or choice variables. Complex state or choice variables are all other state
      variables. For simple state variables it is thus enough to store the grid of
      feasible values.

    Future compressions could be:

    - Use that there are typically only few different choice sets. Thus the state
      choice space could be compressed by only saving the index in a list of choice sets
      and not the entire choice set

    """
    simple_variables, complex_variables = _find_simple_and_complex_variables(model)
    grids = _create_grids(model)

    space = {
        "complex": _create_complex_state_choice_space(
            grids, complex_variables, model.get("filters", [])
        ),
        "simple": _create_simple_state_choice_space(grids, simple_variables),
    }
    # to-do: We probably also need an indexer for the complex state space
    return space


def _find_simple_and_complex_variables(model):
    state_variables = list(model["states"])
    discrete_choices = [
        name for name, spec in model["choices"].items() if "options" in spec
    ]
    all_variables = set(state_variables + discrete_choices)

    # to-do: all dependencies of filtered variables are also a filtered variable
    filtered_variables = {}
    for func in model.get("state_filters", []):
        filtered_variables = filtered_variables.union(
            inspect.signature(func).parameters
        )

    simple_variables = all_variables.difference(filtered_variables)
    complex_variables = all_variables.difference(simple_variables)
    return simple_variables, complex_variables


def _create_grids(model):
    gridspecs = {
        **model["choices"],
        **model["states"],
    }
    grids = {}
    for name, spec in gridspecs.items():
        if "options" in spec:
            grids["name"] = np.array(spec["options"])
        else:
            spec = spec.copy()
            func = getattr(grids_module, spec.pop("grid_type"))
            grids[name] = func(**spec)

    return grids


def _create_complex_state_choice_space(grids, complex_variables, filters):  # noqa: U100
    """Create the ore state choice space.

    Args:
        grids (dict): Dictionary of grids for all variables in the model
        complex_variables (set): Names of the complex variablse
        filters (list): List of filter functions. A filter function depends on one or
            more variables and returns True if a state is feasible.

    Returns:
        dict: Dictionary of arrays where each array represents a column of the core
            state_choice_space.

    """
    if complex_variables or filters:
        # to-do: create state space and apply filters
        raise NotImplementedError()
    else:
        out = {}
    return out


def _create_simple_state_choice_space(grids, simple_variables):
    return {name: grid for name, grid in grids.items() if name in simple_variables}
