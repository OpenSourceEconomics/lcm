import inspect

import jax.numpy as jnp
from dags import concatenate_functions

from lcm.dispatchers import vmap_1d
from lcm.interfaces import Space

# ======================================================================================
# Data State Choice Space
# ======================================================================================


def create_data_state_choice_space(
    initial_states,
    model,
):
    # preparations
    # ==================================================================================
    vi = model.variable_info

    has_sparse_choice_vars = len(vi.query("is_sparse & is_choice")) > 0

    # check that all states have an initial value
    # ==================================================================================
    state_names = set(vi.query("is_state").index)

    if state_names != set(initial_states.keys()):
        raise ValueError(
            "You need to provide an initial value for each state variable in the model."
            f" Missing initial states: {state_names - set(initial_states.keys())}",
        )

    # get sparse and dense choices
    # ==================================================================================
    sparse_choices = {
        name: grid
        for name, grid in model.grids.items()
        if name in vi.query("is_sparse & is_choice").index.tolist()
    }

    dense_choices = {
        name: grid
        for name, grid in model.grids.items()
        if name in vi.query("is_dense & is_choice").index.tolist()
    }

    # create sparse choice state product
    # ==================================================================================
    if has_sparse_choice_vars:
        # create sparse choice product
        # ==============================================================================
        sc_product, n_sc_product_combinations = dict_product(sparse_choices)

        n_initial_states = len(list(initial_states.values())[0])

        # create full sparse choice state product
        # ==============================================================================
        _combination_grid = {}
        for name, state in initial_states.items():
            _combination_grid[name] = jnp.tile(state, reps=n_sc_product_combinations)

        for name, choice in sc_product.items():
            _combination_grid[name] = jnp.repeat(choice, repeats=n_initial_states)

        # create filter mask
        # ==============================================================================
        filter_names = model.function_info.query("is_filter").index.tolist()

        scalar_filter = concatenate_functions(
            functions=model.functions,
            targets=filter_names,
            aggregator=jnp.logical_and,
        )

        parameters = list(inspect.signature(scalar_filter).parameters)
        kwargs = {
            key: val for key, val in _combination_grid.items() if key in parameters
        }

        _filter = vmap_1d(scalar_filter, variables=parameters)
        mask = _filter(**kwargs)

        # filter infeasible combinations
        # ==============================================================================
        combination_grid = {
            name: grid[mask] for name, grid in _combination_grid.items()
        }

    else:
        combination_grid = initial_states

    return Space(sparse_vars=combination_grid, dense_vars=dense_choices)


# ======================================================================================
# Next state
# ======================================================================================


def get_next_state_function(model):
    """Combine the next state functions into one function.

    Args:
        model (Model): Model instance.

    Returns:
        function: Combined next state function.

    """
    targets = model.function_info.query("is_next").index.tolist()

    return concatenate_functions(
        functions=model.functions,
        targets=targets,
        return_type="dict",
    )


# ======================================================================================
# Auxiliary
# ======================================================================================


def dict_product(d):
    """Create a product of the entries of a dictionary.

    Args:
        d (dict): Dictionary where all values are arrays.

    Returns:
        - dict: Dictionary with same keys but values correspond to rows of product.
        - int: Number of all combinations.

    """
    arrays = list(d.values())
    grid = jnp.meshgrid(*arrays, indexing="ij")
    stacked = jnp.stack(grid, axis=-1).reshape(-1, len(arrays))
    return dict(zip(d.keys(), list(stacked.T), strict=True)), len(stacked)
