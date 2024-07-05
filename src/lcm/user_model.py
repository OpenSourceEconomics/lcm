from collections.abc import Callable
from dataclasses import (
    KW_ONLY,
    InitVar,
    dataclass,
    field,
)
from dataclasses import (
    replace as dataclasses_replace,
)
from typing import Any, Literal, NamedTuple, get_args

import jax.numpy as jnp

from lcm.interfaces import (
    ContinuousGridInfo,
    ContinuousGridSpec,
    ContinuousGridType,
    DiscreteGridSpec,
    GridSpec,
)
from lcm.typing import DiscreteLabels, ScalarUserInput

# ======================================================================================
# Errors
# ======================================================================================


class LcmModelInitializationError(Exception):
    """Raised when there is an error in the model initialization."""


class LcmGridInitializationError(Exception):
    """Raised when there is an error in the grid initialization."""


def _format_errors(errors: list[str]) -> str:
    if len(errors) == 1:
        formatted = errors[0]
    else:
        formatted = "\n\n".join([f"{i}. {error}" for i, error in enumerate(errors, 1)])
    return formatted


# ======================================================================================
# User interface
# ======================================================================================


class _ContinuousGridTypeSelector(NamedTuple):
    linspace: ContinuousGridType = "linspace"
    logspace: ContinuousGridType = "logspace"


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
    choices: dict[str, GridSpec] = field(default_factory=dict)
    states: dict[str, GridSpec] = field(default_factory=dict)
    _skip_checks: InitVar[bool] = False

    def __post_init__(self, _skip_checks: bool) -> None:  # noqa: C901
        """Perform basic checks on the user model before the model processing starts."""
        if _skip_checks:
            return

        errors = []

        if (check := n_periods_is_positive_int(self.n_periods)).failed:
            errors.append(check.msg)

        if (check := is_dictionary(self.functions, msg_prefix="functions")).failed:
            errors.append(check.msg)

        if (check := is_dictionary(self.choices, msg_prefix="choices")).failed:
            errors.append(check.msg)

        if (check := is_dictionary(self.states, msg_prefix="states")).failed:
            errors.append(check.msg)

        if (
            check := each_state_or_choice_is_a_valid_lcm_grid(self.states, "state")
        ).failed:
            errors.append(check.msg)

        if (
            check := each_state_or_choice_is_a_valid_lcm_grid(self.choices, "choice")
        ).failed:
            errors.append(check.msg)

        if (
            check := states_and_choices_are_non_overlapping(self.states, self.choices)
        ).failed:
            errors.append(check.msg)

        if (check := utility_function_is_defined(self.functions)).failed:
            errors.append(check.msg)

        if (
            check := each_state_has_a_next_function(self.states, self.functions)
        ).failed:
            errors.append(check.msg)

        if errors:
            raise LcmModelInitializationError(_format_errors(errors))

    def replace(self, **kwargs) -> "Model":
        """Replace the attributes of the model.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the model.

        Returns:
            A new model with the replaced attributes.

        """
        return dataclasses_replace(self, **kwargs)


class Grid:
    """A user interface for creating grids.

    Attributes:
        type: A class attribute that provides access to the grid type selector.

    Methods:
        discrete: A class method for constructing a discrete grid.
        continuous: A class method for constructing a continuous grid.

    """

    type: _ContinuousGridTypeSelector = _ContinuousGridTypeSelector()

    @classmethod
    def discrete(cls, options: DiscreteLabels) -> DiscreteGridSpec:
        """Construct a discrete grid.

        Args:
            options: A list of options for the discrete grid.

        Returns:
            A dictionary with the options for the discrete grid.

        """
        errors = []

        if (check := is_list_or_tuple(options)).failed:
            errors.append(check.msg)

        if (check := all_elements_are_numerical_scalar(options)).failed:
            errors.append(check.msg)

        if errors:
            raise LcmGridInitializationError(_format_errors(errors))

        return jnp.array(options)

    @classmethod
    def continuous(
        cls,
        start: ScalarUserInput,
        stop: ScalarUserInput,
        n_points: int,
        grid_type: ContinuousGridType,
    ) -> ContinuousGridSpec:
        """Construct a continuous grid.

        Args:
            start: The start value for the continuous grid.
            stop: The stop value for the continuous grid.
            n_points: The number of points in the continuous grid.
            grid_type: The type of the continuous grid.

        Returns:
            A dictionary with the start, stop, number of points, and grid type for the
            continuous grid.

        """
        errors = []

        if (check := grid_type_is_valid(grid_type)).failed:
            errors.append(check.msg)

        if (check := is_numerical_scalar(start, "start")).failed:
            errors.append(check.msg)

        if (check := is_numerical_scalar(stop, "stop")).failed:
            errors.append(check.msg)

        if (check := start_is_smaller_than_stop(start, stop)).failed:
            errors.append(check.msg)

        if (check := n_grid_points_is_positive_int(n_points)).failed:
            errors.append(check.msg)

        if errors:
            raise LcmGridInitializationError(_format_errors(errors))

        grid_info = ContinuousGridInfo(
            start=start,
            stop=stop,
            n_points=n_points,
        )
        return ContinuousGridSpec(kind=grid_type, info=grid_info)


# ======================================================================================
# Basic checks for model consistency before the actual model processing starts
# --------------------------------------------------------------------------------------
# Each test function returns a _CheckResult (NamedTuple) object, where the first entry
# (failed: bool) indicates if the test passed and the second entry (msg: str) returns
# information about the test failure.
# ======================================================================================


class _CheckResult(NamedTuple):
    failed: bool
    msg: str


def utility_function_is_defined(functions: dict[str, Callable]) -> _CheckResult:
    msg = (
        "Utility function is not defined. LCM expects a function called 'utility' in "
        "the functions dictionary."
    )
    return _CheckResult(failed="utility" not in functions, msg=msg)


def is_dictionary(
    obj: Any,  # noqa: ANN401
    msg_prefix: str,
) -> _CheckResult:
    msg = f"{msg_prefix} must be a dictionary, but is: {type(obj)}."
    return _CheckResult(failed=not isinstance(obj, dict), msg=msg)


def each_state_or_choice_is_a_valid_lcm_grid(
    states_or_choices_obj: dict[str, GridSpec],
    state_or_choice: Literal["state", "choice"],
) -> _CheckResult:
    # This is checked by `states_or_choices_are_dictionaries`, but to collect as many
    # errors as possible we run all checks even if one fails.
    if not isinstance(states_or_choices_obj, dict):
        return _CheckResult(failed=True, msg="")

    invalid = []
    for key, val in states_or_choices_obj.items():
        if not isinstance(val, get_args(GridSpec)):
            invalid.append(key)
    msg = (
        f"The following {state_or_choice}s are not valid LCM grids: {invalid}. Please "
        "use the class `lcm.Grid` to create valid LCM grids."
    )
    return _CheckResult(failed=bool(invalid), msg=msg)


def states_and_choices_are_non_overlapping(
    states: dict[str, GridSpec],
    choices: dict[str, GridSpec],
) -> _CheckResult:
    # This is checked by `states_or_choices_are_dictionaries`, but to collect as many
    # errors as possible we run all checks even if one fails.
    if not isinstance(states, dict) or not isinstance(choices, dict):
        return _CheckResult(failed=True, msg="")

    overlap = set(states) & set(choices)
    msg = (
        "States and choices cannot have overlapping names. The following names are "
        f"used in both states and choices: {overlap}."
    )
    return _CheckResult(failed=bool(overlap), msg=msg)


def each_state_has_a_next_function(
    states: dict[str, GridSpec],
    functions: dict[str, Callable],
) -> _CheckResult:
    states_without_next_func = [
        state for state in states if f"next_{state}" not in functions
    ]
    msg = (
        "Each state must have a corresponding next state function. For the following "
        f"states, no next state function was found: {states_without_next_func}."
    )
    return _CheckResult(failed=bool(states_without_next_func), msg=msg)


def n_periods_is_positive_int(n_periods: int) -> _CheckResult:
    msg = f"Number of periods must be a positive integer, but is: {n_periods}."
    return _CheckResult(
        failed=not isinstance(n_periods, int) or n_periods <= 0,
        msg=msg,
    )


def is_list_or_tuple(obj: Any) -> _CheckResult:  # noqa: ANN401
    msg = f"Options must be a list or tuple, but is: {type(obj)}."
    return _CheckResult(failed=not isinstance(obj, list | tuple), msg=msg)


def all_elements_are_numerical_scalar(options: list[Any]) -> _CheckResult:
    # This is checked by `is_list_or_tuple`, but to collect as many errors as
    # possible we run all checks even if one fails.
    if not isinstance(options, list | tuple):
        return _CheckResult(failed=True, msg="")

    msg = (
        "Options must be numerical scalars (int, float, 0-dimensional jax.Array), but "
        f"are: {options}."
    )
    return _CheckResult(
        failed=any(
            not isinstance(option, get_args(ScalarUserInput)) for option in options
        ),
        msg=msg,
    )


def grid_type_is_valid(grid_type: ContinuousGridType) -> _CheckResult:
    msg = (
        f"Grid type must be either 'linspace' or 'logspace', but is: '{grid_type}'. "
        "Use `lcm.Grid.type.` to see a selection of valid grid types."
    )
    return _CheckResult(failed=grid_type not in get_args(ContinuousGridType), msg=msg)


def n_grid_points_is_positive_int(n_points: int) -> _CheckResult:
    msg = f"Number of grid points must be a positive integer, but is: {n_points}."
    return _CheckResult(failed=not isinstance(n_points, int) or n_points <= 0, msg=msg)


def is_numerical_scalar(obj: Any, msg_prefix: str) -> _CheckResult:  # noqa: ANN401
    msg = f"{msg_prefix} must be a numerical scalar, but is: {type(obj)}."
    return _CheckResult(failed=not isinstance(obj, get_args(ScalarUserInput)), msg=msg)


def start_is_smaller_than_stop(
    start: ScalarUserInput,
    stop: ScalarUserInput,
) -> _CheckResult:
    # This is checked by `is_numerical_scalar`, but to collect as many errors as
    # possible we run all checks even if one fails.
    if not isinstance(start, get_args(ScalarUserInput)) or not isinstance(
        stop,
        get_args(ScalarUserInput),
    ):
        return _CheckResult(failed=True, msg="")

    msg = f"Start must be smaller than stop, but is: start={start} and stop={stop}."
    return _CheckResult(failed=start >= stop, msg=msg)
