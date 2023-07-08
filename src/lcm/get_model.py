"""Get a user model and parameters."""
from typing import NamedTuple

from lcm.example_models import (
    PHELPS_DEATON,
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


ISKHAKOV_2017_FIVE_PERIODS = {
    **PHELPS_DEATON,
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

# ======================================================================================
# Model collection
# ======================================================================================

MODELS = {
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
}
