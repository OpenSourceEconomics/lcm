import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
from dags.signature import with_signature
from jax import Array

from lcm.functools import all_as_kwargs
from lcm.grids import DiscreteGrid
from lcm.input_processing.util import (
    get_gridspecs,
)
from lcm.typing import ParamsDict
from lcm.user_model import Model


@dataclass
class Converter:
    """Converts between representations of discrete states and their parameters.

    While LCM supports general discrete grids, internally, these are converted to
    indices. This class provides functionality for converting between the internal
    representation and the external representation.

    Attributes:
        converted_states: The names of the states that have been converted.
        index_to_label: A dictionary of functions mapping from the internal index to the
            label for each converted state.
        label_to_index: A dictionary of functions mapping from the label to the internal
            index for each converted state.

    """

    converted_states: list[str] = field(default_factory=list)
    index_to_label: dict[str, Callable[[Array], Array]] = field(default_factory=dict)
    label_to_index: dict[str, Callable[[Array], Array]] = field(default_factory=dict)

    def params_from_internal(self, params: ParamsDict) -> ParamsDict:
        """Convert parameters from internal to external representation.

        If a state has been converted, the name of its corresponding next function must
        be changed from `next___{var}_index__` to `next_{var}`.

        """
        out = params.copy()
        for var in self.converted_states:
            out.pop(f"next___{var}_index__")
            out[f"next_{var}"] = params[f"next___{var}_index__"]
        return out

    def params_to_internal(self, params: ParamsDict) -> ParamsDict:
        """Convert parameters from external to internal representation.

        If a state has been converted, the name of its corresponding next function must
        be changed from `next_{var}` to `next___{var}_index__`.

        """
        out = params.copy()
        for var in self.converted_states:
            out.pop(f"next_{var}")
            out[f"next___{var}_index__"] = params[f"next_{var}"]
        return out

    def states_from_internal(self, states: dict[str, Array]) -> dict[str, Array]:
        """Convert states from internal to external representation.

        If a state has been converted, the name of its corresponding index function must
        be changed from `___{var}_index__` to `{var}`, and the values of the state must
        be converted from indices to labels.

        """
        out = states.copy()
        for var in self.converted_states:
            out.pop(f"__{var}_index__")
            out[var] = self.index_to_label[var](states[f"__{var}_index__"])
        return out

    def states_to_internal(self, states: dict[str, Array]) -> dict[str, Array]:
        """Convert states from external to internal representation.

        If a state has been converted, the name of its corresponding index function must
        be changed from `{var}` to `___{var}_index__`, and the values of the state must
        be converted from labels to indices.

        """
        out = states.copy()
        for var in self.converted_states:
            out.pop(var)
            out[f"__{var}_index__"] = self.label_to_index[var](states[var])
        return out


def convert_discrete_options_to_indices(
    model: Model,
) -> tuple[Model, Converter]:
    """Update the user model to ensure that discrete variables have index options.

    For each discrete variable with non-index options, we:

        1. Remove the variable from the states or choices dictionary
        2. Replace it with a new state or choice with index options (__{var}_index__)
        3. Add a function that maps the index options to the original options
        4. Add updated next functions (if the variable was a state variable)

    Args:
        model: The model as provided by the user.

    Returns:
        - The model with all discrete variables having index options.
        - A converter that can be used to convert between the internal and external
          representation of the model.

    """
    gridspecs = get_gridspecs(model)

    non_index_discrete_vars = _get_discrete_vars_with_non_index_options(model)

    if not non_index_discrete_vars:
        # fast path
        return model, Converter()

    functions = model.functions.copy()
    states = model.states.copy()
    choices = model.choices.copy()

    # Update next functions (needs to be done before updating the grids, otherwise the
    # already updated state variables are being used)
    # ----------------------------------------------------------------------------------
    non_index_states = [s for s in states if s in non_index_discrete_vars]

    for state in model.states:
        next_func = model.functions[f"next_{state}"]
        must_be_updated = _func_depends_on(next_func, depends_on=non_index_states)
        if must_be_updated:
            functions.pop(f"next_{state}")

            functions[f"next___{state}_index__"] = _get_next_func_of_index_var(
                next_func=next_func,
                variables=non_index_states,
            )

    # Update grids
    # ----------------------------------------------------------------------------------
    for var in non_index_discrete_vars:
        grid: DiscreteGrid = gridspecs[var]  # type: ignore[assignment]
        index_grid = DiscreteGrid(options=list(range(len(grid.options))))

        if var in states:
            states.pop(var)
            states[f"__{var}_index__"] = index_grid
        else:
            choices.pop(var)
            choices[f"__{var}_index__"] = index_grid

    # Add index to label functions
    # ----------------------------------------------------------------------------------
    index_to_label_funcs = {
        var: _get_index_to_label_func(gridspecs[var].to_jax(), name=var)
        for var in non_index_discrete_vars
    }
    functions = functions | index_to_label_funcs

    # Construct label to index functions for states
    # ----------------------------------------------------------------------------------
    converted_states = [s for s in non_index_discrete_vars if s in model.states]

    label_to_index_funcs_for_states = {
        var: _get_label_to_index_func(gridspecs[var].to_jax(), name=var)
        for var in converted_states
    }

    # Subset index to label functions to only include states for converter
    index_to_label_funcs_for_states = {
        k: v for k, v in index_to_label_funcs.items() if k in model.states
    }

    converter = Converter(
        converted_states=converted_states,
        index_to_label=index_to_label_funcs_for_states,
        label_to_index=label_to_index_funcs_for_states,
    )

    new_model = model.replace(
        states=states,
        choices=choices,
        functions=functions,
    )
    return new_model, converter


def _get_next_func_of_index_var(next_func: Callable, variables: list[str]) -> Callable:
    """Create next function for corresponding index variable."""
    arg_names = list(inspect.signature(next_func).parameters)

    relevant_vars = [var for var in variables if var in arg_names]

    if not relevant_vars:
        return next_func

    for var in relevant_vars:
        arg_names[arg_names.index(var)] = f"__{var}_index__"

    @with_signature(args=arg_names)
    def next_func_of_index_var(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=arg_names)
        for var in relevant_vars:
            kwargs[var] = kwargs.pop(f"__{var}_index__")
        return next_func(**kwargs)

    return next_func_of_index_var


def _get_discrete_vars_with_non_index_options(model: Model) -> list[str]:
    """Get discrete variables with non-index options.

    Collect all discrete variables with options that do not correspond to indices.

    """
    gridspecs = get_gridspecs(model)
    discrete_vars = []
    for name, spec in gridspecs.items():
        if isinstance(spec, DiscreteGrid) and list(spec.options) != list(
            range(len(spec.options))
        ):
            discrete_vars.append(name)
    return discrete_vars


def _get_index_to_label_func(
    labels_array: Array, name: str
) -> Callable[[Array], Array]:
    """Get function mapping from index to label.

    Args:
        labels_array: An array of labels.
        name: The name of resulting function argument.

    Returns:
        A function mapping an array with indices corresponding to labels_array to the
        corresponding labels.

    """
    arg_name = f"__{name}_index__"

    @with_signature(args=[arg_name])
    def func(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=[arg_name])
        index = kwargs[arg_name]
        return labels_array[index]

    return func


def _get_label_to_index_func(
    labels_array: Array, name: str
) -> Callable[[Array], Array]:
    """Get function mapping from label to index.

    Args:
        labels_array: An array of labels.
        name: The name of resulting function argument.

    Returns:
        A function mapping an array with values in labels_array to their corresponding
        indices.

    """

    @with_signature(args=[name])
    def label_to_index(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=[name])
        data = kwargs[name]
        return jnp.argmax(data[:, None] == labels_array[None, :], axis=1)

    return label_to_index


def _func_depends_on(func: Callable, depends_on: list[str]) -> bool:
    """Check if any function argument is in the list depends_on."""
    arg_names = list(inspect.signature(func).parameters)
    return any(arg in depends_on for arg in arg_names)
