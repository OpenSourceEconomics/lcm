import numpy as np
from lcm.create_params import (
    _create_function_params,
    _create_shock_params,
    create_params,
)
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS
from pybaum import leaf_names


def test_create_params_phelps_deaton_with_shocks():
    params = create_params(PHELPS_DEATON_WITH_SHOCKS)

    names = leaf_names(params, separator="$")
    expected_names = [
        "beta",
        "utility$delta",
        "next_wealth$interest_rate",
        "next_wealth$wage",
        "wage_shock$sd",
        "additive_utility_shock$scale",
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


def test_create_shock_params():
    shocks = {
        "a": "lognormal",
        "b": "extreme_value",
    }
    got = _create_shock_params(shocks)

    assert {"a", "b"} == set(got.keys())
    assert got["a"] == {"sd": np.nan}
    assert got["b"] == {"scale": np.nan}
