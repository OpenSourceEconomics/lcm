import pytest

from lcm.exceptions import ModelInitilizationError
from lcm.grids import DiscreteGrid
from lcm.model import Model


def test_model_invalid_states():
    with pytest.raises(ModelInitilizationError, match="states must be a dictionary"):
        Model(
            n_periods=2,
            states="health",
            choices={},
            functions={"utility": lambda: 0},
        )


def test_model_invalid_choices():
    with pytest.raises(ModelInitilizationError, match="choices must be a dictionary"):
        Model(
            n_periods=2,
            states={},
            choices="exercise",
            functions={"utility": lambda: 0},
        )


def test_model_invalid_functions():
    with pytest.raises(ModelInitilizationError, match="functions must be a dictionary"):
        Model(
            n_periods=2,
            states={},
            choices={},
            functions="utility",
        )


def test_model_invalid_functions_values():
    with pytest.raises(
        ModelInitilizationError, match="function values must be a callable, but is 0."
    ):
        Model(
            n_periods=2,
            states={},
            choices={},
            functions={"utility": 0},
        )


def test_model_invalid_functions_keys():
    with pytest.raises(
        ModelInitilizationError, match="function keys must be a strings, but is 0."
    ):
        Model(
            n_periods=2,
            states={},
            choices={},
            functions={0: lambda: 0},
        )


def test_model_invalid_choices_values():
    with pytest.raises(
        ModelInitilizationError, match="choices value 0 must be an LCM grid."
    ):
        Model(
            n_periods=2,
            states={},
            choices={"exercise": 0},
            functions={"utility": lambda: 0},
        )


def test_model_invalid_states_values():
    with pytest.raises(
        ModelInitilizationError, match="states value 0 must be an LCM grid."
    ):
        Model(
            n_periods=2,
            states={"health": 0},
            choices={},
            functions={"utility": lambda: 0},
        )


def test_model_invalid_n_periods():
    with pytest.raises(
        ModelInitilizationError, match="Number of periods must be a positive integer."
    ):
        Model(
            n_periods=0,
            states={},
            choices={},
            functions={"utility": lambda: 0},
        )


def test_model_missing_next_func():
    with pytest.raises(
        ModelInitilizationError,
        match="Each state must have a corresponding next state function.",
    ):
        Model(
            n_periods=2,
            states={"health": DiscreteGrid([0, 1])},
            choices={"exercise": DiscreteGrid([0, 1])},
            functions={"utility": lambda: 0},
        )


def test_model_missing_utility():
    with pytest.raises(
        ModelInitilizationError,
        match="Utility function is not defined. LCM expects a function called 'utility",
    ):
        Model(
            n_periods=2,
            states={},
            choices={},
            functions={},
        )


def test_model_overlapping_states_choices():
    with pytest.raises(
        ModelInitilizationError,
        match="States and choices cannot have overlapping names.",
    ):
        Model(
            n_periods=2,
            states={"health": DiscreteGrid([0, 1])},
            choices={"health": DiscreteGrid([0, 1])},
            functions={"utility": lambda: 0},
        )
