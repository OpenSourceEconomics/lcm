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


MODELS = {
    "iskhakov_2017_test": ModelAndParams(
        model={**PHELPS_DEATON, "n_periods": 5},
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
