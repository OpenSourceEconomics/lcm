from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
from dags.signature import with_signature
from jax import Array

from lcm.functools import all_as_kwargs
from lcm.typing import ParamsDict


@dataclass
class Converter:
    converted_states: list[str] = field(default_factory=list)
    index_to_label: dict[str, Callable[[Array], Array]] = field(default_factory=dict)
    label_to_index: dict[str, Callable[[Array], Array]] = field(default_factory=dict)

    def params_from_internal(self, params: ParamsDict) -> ParamsDict:
        out = params.copy()
        for var in self.converted_states:
            out.pop(f"next___{var}_index__")
            out[var] = params[f"next___{var}_index__"]
        return out

    def params_to_internal(self, params: ParamsDict) -> ParamsDict:
        out = params.copy()
        for var in self.converted_states:
            out.pop(var)
            out[f"next___{var}_index__"] = params[var]
        return out

    def states_from_internal(self, states: dict[str, Array]) -> dict[str, Array]:
        out = states.copy()
        for var in self.converted_states:
            out.pop(f"__{var}_index__")
            out[var] = self.index_to_label[var](states[f"__{var}_index__"])
        return out

    def states_to_internal(self, states: dict[str, Array]) -> dict[str, Array]:
        out = states.copy()
        for var in self.converted_states:
            out.pop(var)
            out[f"__{var}_index__"] = self.label_to_index[var](states[var])
        return out


def get_label_to_index_func(labels_array, name):
    @with_signature(args=[name])
    def label_to_index(*args, **kwargs):
        kwargs = all_as_kwargs(args, kwargs, arg_names=[name])
        data = kwargs[name]
        indices = jnp.argmax(data[:, None] == labels_array[None, :], axis=1)
        return indices

    return label_to_index
