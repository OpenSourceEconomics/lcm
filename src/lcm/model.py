from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import NamedTuple, TypedDict, get_args

from lcm.interfaces import ContinuousGridType
from lcm.typing import DiscreteLabels, ScalarUserInput

# ======================================================================================
# Errors
# ======================================================================================


class LcmModelInitializationError(Exception):
    """Raised when there is an error in the model initialization."""


def _format_errors(errors: list[str]) -> str:
    if len(errors) == 1:
        formatted = errors[0]
    else:
        formatted = "\n\n".join([f"{i}. {error}" for i, error in enumerate(errors, 1)])
    return formatted


# ======================================================================================
# Grid Types
# ======================================================================================


class _DiscreteUserGrid(TypedDict):
    options: DiscreteLabels


class _ContinuousUserGrid(TypedDict):
    start: ScalarUserInput
    stop: ScalarUserInput
    n_points: int
    grid_type: ContinuousGridType


class _ContinuousGridTypeSelector(NamedTuple):
    linspace: ContinuousGridType = "linspace"
    logspace: ContinuousGridType = "logspace"


# ======================================================================================
# User interface
# ======================================================================================


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
    functions: dict[str, Callable]
    choices: dict[str, _DiscreteUserGrid | _ContinuousUserGrid]
    states: dict[str, _DiscreteUserGrid | _ContinuousUserGrid]

    def __post_init__(self) -> None:
        """Perform basic checks on the user model before the model processing starts."""
        errors = []

        if (check := utility_function_is_defined(self.functions)).failed:
            errors.append(check.msg)

        if (
            check := each_state_has_a_next_function(self.states, self.functions)
        ).failed:
            errors.append(check.msg)

        if (check := n_periods_is_positive_int(self.n_periods)).failed:
            errors.append(check.msg)

        if errors:
            raise LcmModelInitializationError(_format_errors(errors))


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
    def discrete(cls, options: DiscreteLabels) -> _DiscreteUserGrid:
        """Construct a discrete grid.

        Args:
            options: A list of options for the discrete grid.

        Returns:
            A dictionary with the options for the discrete grid.

        """
        if (check := options_are_numerical_scalar(options)).failed:
            raise LcmModelInitializationError(check.msg)

        return {"options": options}

    @classmethod
    def continuous(
        cls,
        start: ScalarUserInput,
        stop: ScalarUserInput,
        n_points: int,
        grid_type: ContinuousGridType,
    ) -> _ContinuousUserGrid:
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

        if (check := start_is_smaller_than_stop(start, stop)).failed:
            errors.append(check.msg)

        if (check := n_grid_points_is_positive_int(n_points)).failed:
            errors.append(check.msg)

        if errors:
            raise LcmModelInitializationError(_format_errors(errors))

        return {
            "start": start,
            "stop": stop,
            "n_points": n_points,
            "grid_type": grid_type,
        }


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


def each_state_has_a_next_function(
    states: dict[str, _DiscreteUserGrid | _ContinuousUserGrid],
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


def options_are_numerical_scalar(options: DiscreteLabels) -> _CheckResult:
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


def start_is_smaller_than_stop(
    start: ScalarUserInput,
    stop: ScalarUserInput,
) -> _CheckResult:
    msg = f"Start must be smaller than stop, but is: start={start} and stop={stop}."
    return _CheckResult(failed=start >= stop, msg=msg)
