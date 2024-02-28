"""Get a user model and parameters."""

from typing import NamedTuple

from pybaum import tree_update

from tests.test_models.phelps_deaton import (
    PHELPS_DEATON,
    PHELPS_DEATON_WITH_FILTERS,
)


class ModelAndParams(NamedTuple):
    """Model and parameters."""

    model: dict
    params: dict


def get_model(model: str):
    """Get a user model and parameters.

    Args:
        model (str): Model name.

    Returns:
        NamedTuple: Model and parameters. Has attributes `model` and `params`.

    """
    if model not in MODELS:
        raise ValueError(f"Model {model} not found. Choose from {set(MODELS.keys())}.")
    return MODELS[model]


# ======================================================================================
# Models
# ======================================================================================

# Remove age and wage functions from Phelps-Deaton model, as they are not used in the
# original paper.
PHELPS_DEATON_WITHOUT_AGE = PHELPS_DEATON.copy()
PHELPS_DEATON_WITHOUT_AGE["functions"] = {
    name: func
    for name, func in PHELPS_DEATON_WITHOUT_AGE["functions"].items()
    if name not in ["age", "wage"]
}


PHELPS_DEATON_FIVE_PERIODS = {
    **PHELPS_DEATON_WITHOUT_AGE,
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": 500,
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": 100,
        },
    },
    "n_periods": 5,
}


ISKHAKOV_2017_FIVE_PERIODS = {
    **PHELPS_DEATON_WITH_FILTERS,
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": 500,
        },
    },
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": 100,
        },
        "lagged_retirement": {"options": [0, 1]},
    },
    "n_periods": 5,
}


ISKHAKOV_2017_THREE_PERIODS = tree_update(ISKHAKOV_2017_FIVE_PERIODS, {"n_periods": 3})

# ======================================================================================
# Models and params
# ======================================================================================

MODELS = {
    "phelps_deaton_regression_test": ModelAndParams(
        model=PHELPS_DEATON_FIVE_PERIODS,
        params={
            "beta": 1.0,
            "utility": {"delta": 1.0},
            "next_wealth": {
                "interest_rate": 0.05,
                "wage": 1.0,
            },
        },
    ),
    "iskhakov_2017_five_periods": ModelAndParams(
        model=ISKHAKOV_2017_FIVE_PERIODS,
        params={
            "beta": 0.98,
            "utility": {"delta": 1.0},
            "next_wealth": {
                "interest_rate": 0.0,
                "wage": 20.0,
            },
        },
    ),
    "iskhakov_2017_low_delta": ModelAndParams(
        model=ISKHAKOV_2017_THREE_PERIODS,
        params={
            "beta": 0.98,
            "utility": {"delta": 0.1},
            "next_wealth": {
                "interest_rate": 0.0,
                "wage": 20.0,
            },
        },
    ),
}
