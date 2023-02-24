from lcm.create_params import create_params
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS
from pybaum import leaf_names


def test_create_params_phelps_deaton_with_shocks():
    params = create_params(PHELPS_DEATON_WITH_SHOCKS)

    names = leaf_names(params, separator="__")
    expected_names = [
        "beta",
        "utility__delta",
        "next_wealth__interest_rate",
        "next_wealth__wage",
        "wage_shock__sd",
        "additive_utility_shock__scale",
    ]

    assert sorted(names) == sorted(expected_names)
