from lcm.create_params import create_params
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
