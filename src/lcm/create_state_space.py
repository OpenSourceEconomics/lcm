"""Create a state space for a given model."""
import numpy as np
from dags import get_ancestors
from lcm import grids as grids_module


def create_state_choice_space(model):
    """Create a state choice space for the model.

    A state_choice_space is a compressed representation of all feasible states and the
    feasible choices within that state. We currently use the following compressions:

    We distinguish between dense and sparse variables (dense_vars and sparse_vars).
    Dense state or choice variables are those whose set of feasible values does not
    depend on any other state or choice variables. Sparse state or choice variables are
    all other state variables. For dense state variables it is thus enough to store the
    grid of feasible values (value_grid), whereas for sparse variables all feasible
    combinations (combination_grid) have to be stored.

    """
    dense_vars, sparse_vars = _find_dense_and_sparse_variables(model)
    grids = _create_grids_from_gridspecs(model)

    space = {
        "combination_grid": _create_combination_grid(
            grids, sparse_vars, model.get("filters", [])
        ),
        "value_grid": _create_value_grid(grids, dense_vars),
    }
    return space


def _find_dense_and_sparse_variables(model):
    state_variables = list(model["states"])
    discrete_choices = [
        name for name, spec in model["choices"].items() if "options" in spec
    ]
    all_variables = set(state_variables + discrete_choices)

    filtered_variables = {}
    filters = model.get("state_filters", [])
    for func in filters:
        filtered_variables = filtered_variables.union(
            get_ancestors(filters, func.__name__)
        )

    dense_vars = all_variables.difference(filtered_variables)
    sparse_vars = all_variables.difference(dense_vars)
    return dense_vars, sparse_vars


def _create_grids_from_gridspecs(model):
    gridspecs = {
        **model["choices"],
        **model["states"],
    }
    grids = {}
    for name, spec in gridspecs.items():
        if "options" in spec:
            grids[name] = np.array(spec["options"])
        else:
            spec = spec.copy()
            func = getattr(grids_module, spec.pop("grid_type"))
            grids[name] = func(**spec)

    return grids


def _create_combination_grid(grids, subset, filters):  # noqa: U100
    """Create the ore state choice space.

    Args:
        grids (dict): Dictionary of grids for all variables in the model
        sparse_vars (set): Names of the sparse_variables
        filters (list): List of filter functions. A filter function depends on one or
            more variables and returns True if a state is feasible.

    Returns:
        dict: Dictionary of arrays where each array represents a column of the core
            state_choice_space.

    """
    if subset or filters:
        # to-do: create state space and apply filters
        raise NotImplementedError()
    else:
        out = {}
    return out


def _create_value_grid(grids, subset):
    return {name: grid for name, grid in grids.items() if name in subset}
