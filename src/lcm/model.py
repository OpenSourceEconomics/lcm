"""Collection of classes that are used by the user to define the model and grids."""

import dataclasses as dc
from collections.abc import Callable
from dataclasses import KW_ONLY, InitVar, dataclass, field

from lcm.exceptions import ModelInitilizationError, format_messages
from lcm.grids import Grid


@dataclass(frozen=True)
class Model:
    """A user model which can be processed into an internal model.

    Attributes:
        description: Description of the model.
        n_periods: Number of periods in the model.
        functions: Dictionary of user provided functions that define the functional
            relationships between model variables. It must include at least a function
            called 'utility'.
        choices: Dictionary of user provided choices.
        states: Dictionary of user provided states.

    """

    description: str | None = None
    _: KW_ONLY
    n_periods: int
    functions: dict[str, Callable] = field(default_factory=dict)
    choices: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    _skip_checks: InitVar[bool] = False

    def __post_init__(self, _skip_checks: bool) -> None:
        if _skip_checks:
            return

        if type_errors := _validate_attribute_types(self):
            msg = format_messages(type_errors)
            raise ModelInitilizationError(msg)

        if logical_errors := _validate_logical_consistency(self):
            msg = format_messages(logical_errors)
            raise ModelInitilizationError(msg)

    def replace(self, **kwargs) -> "Model":
        """Replace the attributes of the model.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the model.

        Returns:
            A new model with the replaced attributes.

        """
        return dc.replace(self, **kwargs)


def _validate_attribute_types(model: Model) -> list[str]:
    """Validate the types of the model attributes."""
    error_messages = []

    # Validate types of states and choices
    # ----------------------------------------------------------------------------------
    for attr_name in ("choices", "states"):
        attr = getattr(model, attr_name)
        if not isinstance(attr, dict):
            error_messages.append(f"{attr_name} must be a dictionary.")
        else:
            for k, v in attr.items():
                if not isinstance(k, str):
                    error_messages.append(f"{attr_name} key {k} must be a string.")
                if not isinstance(v, Grid):
                    error_messages.append(f"{attr_name} value {v} must be an LCM grid.")

    # Validate types of functions
    # ----------------------------------------------------------------------------------
    if not isinstance(model.functions, dict):
        error_messages.append("functions must be a dictionary.")
    else:
        for k, v in model.functions.items():
            if not isinstance(k, str):
                error_messages.append(f"functions key {k} must be a string.")
            if not callable(v):
                error_messages.append(f"functions value {v} must be a callable.")

    return error_messages


def _validate_logical_consistency(model: Model) -> list[str]:
    """Validate the logical consistency of the model."""
    error_messages = []

    if model.n_periods < 1:
        error_messages.append("Number of periods must be a positive integer.")

    if "utility" not in model.functions:
        error_messages.append(
            "Utility function is not defined. LCM expects a function called 'utility'"
            "in the functions dictionary.",
        )

    if states_without_next_func := [
        state for state in model.states if f"next_{state}" not in model.functions
    ]:
        error_messages.append(
            "Each state must have a corresponding next state function. For the "
            "following states, no next state function was found: "
            f"{states_without_next_func}.",
        )

    if states_and_choices_overlap := set(model.states) & set(model.choices):
        error_messages.append(
            "States and choices cannot have overlapping names. The following names "
            f"are used in both states and choices: {states_and_choices_overlap}.",
        )

    return error_messages
