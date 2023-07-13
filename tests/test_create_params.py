import numpy as np
from lcm.create_params import (
    _create_function_params,
    create_params,
)
from lcm.example_models import PHELPS_DEATON_WITH_FILTERS
from pybaum import leaf_names


def test_create_params_phelps_deaton_with_filters():
    params = create_params(PHELPS_DEATON_WITH_FILTERS)

    names = leaf_names(params, separator="__")
    expected_names = [
        "beta",
        "utility__delta",
        "next_wealth__interest_rate",
        "next_wealth__wage",
    ]

    assert sorted(names) == sorted(expected_names)


def test_create_params_without_shocks():
    model = {
        "functions": {
            "f": lambda a, b, c: None,  # noqa: ARG005
        },
        "choices": {
            "a": None,
        },
        "states": {
            "b": None,
        },
    }
    got = create_params(model)
    assert got == {"beta": np.nan, "f": {"c": np.nan}}


def test_create_function_params():
    model = {
        "functions": {
            "f": lambda a, b, c: None,  # noqa: ARG005
        },
        "choices": {
            "a": None,
        },
        "states": {
            "b": None,
        },
    }
    got = _create_function_params(model)
    assert got == {"f": {"c": np.nan}}
