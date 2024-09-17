from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
from dags.signature import with_signature
from jax import Array

from lcm.functools import all_as_kwargs
from lcm.typing import ParamsDict


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


def get_label_to_index_func(labels_array: Array, name: str) -> Callable[[Array], Array]:
    """Get function mapping from label to index.

    Args:
        labels_array: An array of labels.
        name: The name of the label.

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
