from collections.abc import Callable
from dataclasses import dataclass, field, make_dataclass

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


@dataclass(frozen=True)
class DiscreteGridConverter:
    """Converts between representations of discrete variables and their parameters.

    While LCM supports general discrete grids, internally, these are converted to
    indices. This class provides functionality for converting between the internal
    representation and the external representation.

    Attributes:
        index_to_code: A dictionary of functions mapping from the internal index to the
            code for each converted state. Keys correspond to the names of converted
            discrete variables.
        code_to_index: A dictionary of functions mapping from the code to the internal
            index for each converted state. Keys correspond to the names of converted
            discrete variables.

    """

    index_to_code: dict[str, Callable[[Array], Array]] = field(default_factory=dict)
    code_to_index: dict[str, Callable[[Array], Array]] = field(default_factory=dict)

    def internal_to_params(self, params: ParamsDict) -> ParamsDict:
        """Convert parameters from internal to external representation.

        If a state has been converted, the name of its corresponding next function must
        be changed from `next___{var}_index__` to `next_{var}`.

        """
        out = params.copy()
        for var in self.index_to_code:
            out.pop(f"next___{var}_index__")
            out[f"next_{var}"] = params[f"next___{var}_index__"]
        return out

    def params_to_internal(self, params: ParamsDict) -> ParamsDict:
        """Convert parameters from external to internal representation.

        If a state has been converted, the name of its corresponding next function must
        be changed from `next_{var}` to `next___{var}_index__`.

        """
        out = params.copy()
        for var in self.index_to_code:
            out.pop(f"next_{var}")
            out[f"next___{var}_index__"] = params[f"next_{var}"]
        return out

    def internal_to_states(self, states: dict[str, Array]) -> dict[str, Array]:
        """Convert states from internal to external representation.

        If a state has been converted, the name of its corresponding index function must
        be changed from `___{var}_index__` to `{var}`, and the values of the state must
        be converted from indices to codes.

        """
        out = states.copy()
        for var, index_to_code in self.index_to_code.items():
            out.pop(f"__{var}_index__")
            out[var] = index_to_code(states[f"__{var}_index__"])
        return out

    def states_to_internal(self, states: dict[str, Array]) -> dict[str, Array]:
        """Convert states from external to internal representation.

        If a state has been converted, the name of its corresponding index function must
        be changed from `{var}` to `___{var}_index__`, and the values of the state must
        be converted from codes to indices.

        """
        out = states.copy()
        for var, code_to_index in self.code_to_index.items():
            out.pop(var)
            out[f"__{var}_index__"] = code_to_index(states[var])
        return out

    def internal_to_choices(self, choices: dict[str, Array]) -> dict[str, Array]:
        """Convert choices from internal to external representation."""
        out = choices.copy()
        for var, index_to_code in self.index_to_code.items():
            out.pop(f"__{var}_index__")
            out[var] = index_to_code(choices[f"__{var}_index__"])
        return out

    def choices_to_internal(self, choices: dict[str, Array]) -> dict[str, Array]:
        """Convert choices from external to internal representation."""
        out = choices.copy()
        for var, code_to_index in self.code_to_index.items():
            out.pop(var)
            out[f"__{var}_index__"] = code_to_index(choices[var])
        return out


def convert_arbitrary_codes_to_array_indices(
    model: Model,
) -> tuple[Model, DiscreteGridConverter]:
    """Update the user model to ensure that discrete variables have index codes.

    For each discrete variable with non-index codes, we:

        1. Remove the variable from the states or choices dictionary
        2. Replace it with a new state or choice with array index codes
        3. Add updated next functions (if the variable was a state variable)
        4. Add a function that maps the array index codes to the original codes

    Args:
        model: The model as provided by the user.

    Returns:
        - The model with all discrete variables having index codes.
        - A converter that can be used to convert between the internal and external
          representation of the model.

    """
    gridspecs = get_gridspecs(model)

    non_index_discrete_vars = _get_discrete_vars_with_non_index_codes(model)

    # fast path
    if not non_index_discrete_vars:
        return model, DiscreteGridConverter()

    functions = model.functions.copy()
    states = model.states.copy()
    choices = model.choices.copy()

    # Update grids
    # ----------------------------------------------------------------------------------
    for var in non_index_discrete_vars:
        grid: DiscreteGrid = gridspecs[var]  # type: ignore[assignment]
        index_category_class = make_dataclass(
            grid.__str__(),
            [(f"__{name}_index__", int, i) for i, name in enumerate(grid.categories)],
        )
        index_grid = DiscreteGrid(index_category_class)

        if var in model.states:
            states.pop(var)
            states[f"__{var}_index__"] = index_grid
        else:
            choices.pop(var)
            choices[f"__{var}_index__"] = index_grid

    # Update next functions
    # ----------------------------------------------------------------------------------
    non_index_states = [s for s in model.states if s in non_index_discrete_vars]

    for var in non_index_states:
        functions[f"next___{var}_index__"] = functions.pop(f"next_{var}")

    # Add index to code functions
    # ----------------------------------------------------------------------------------
    index_to_code_funcs = {
        var: _get_index_to_code_func(gridspecs[var].to_jax(), name=var)
        for var in non_index_discrete_vars
    }
    functions = functions | index_to_code_funcs

    # Create code to index functions for converter
    # ----------------------------------------------------------------------------------
    code_to_index_funcs = {
        var: _get_code_to_index_func(gridspecs[var].to_jax(), name=var)
        for var in non_index_discrete_vars
    }

    discrete_grid_converter = DiscreteGridConverter(
        index_to_code=index_to_code_funcs,
        code_to_index=code_to_index_funcs,
    )

    new_model = model.replace(
        states=states,
        choices=choices,
        functions=functions,
    )
    return new_model, discrete_grid_converter


def _get_discrete_vars_with_non_index_codes(model: Model) -> list[str]:
    """Get discrete variables with non-index codes.

    Collect all discrete variables with codes that do not correspond to indices.

    """
    gridspecs = get_gridspecs(model)
    discrete_vars = []
    for name, spec in gridspecs.items():
        if isinstance(spec, DiscreteGrid) and list(spec.codes) != list(
            range(len(spec.codes))
        ):
            discrete_vars.append(name)
    return discrete_vars


def _get_index_to_code_func(codes_array: Array, name: str) -> Callable[[Array], Array]:
    """Get function mapping from index to code.

    Args:
        codes_array: An array of codes.
        name: The name of resulting function argument.

    Returns:
        A function mapping an array with indices corresponding to codes_array to the
        corresponding codes.

    """
    arg_name = f"__{name}_index__"

    @with_signature(args=[arg_name])
    def func(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=[arg_name])
        index = kwargs[arg_name]
        return codes_array[index]

    return func


def _get_code_to_index_func(codes_array: Array, name: str) -> Callable[[Array], Array]:
    """Get function mapping from code to index.

    Args:
        codes_array: An array of codes.
        name: The name of resulting function argument.

    Returns:
        A function mapping an array with values in codes_array to their corresponding
        indices.

    """

    @with_signature(args=[name])
    def code_to_index(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=[name])
        data = kwargs[name]
        return jnp.argmax(data[:, None] == codes_array[None, :], axis=1)

    return code_to_index
