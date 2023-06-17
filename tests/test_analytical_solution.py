"""Testing against the analytical solution by Iskhakov et al (2017)."""
import pickle
from pathlib import Path

import numpy as np
import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON_NO_BORROWING
from numpy.testing import assert_array_almost_equal as aaae
from pybaum import tree_update

DATA = Path(__file__).parent.resolve().joinpath("analytical_solution")


@pytest.fixture()
def test_model():
    config_update = {
        "n_periods": 5,
        "choices": {
            "consumption": {
                "n_points": 10_000,
            },
        },
        "states": {
            "wealth": {
                "n_points": 10_000,
            },
        },
    }
    return tree_update(PHELPS_DEATON_NO_BORROWING, config_update)


TEST_CASES = {
    "iskhakov_2017": {
        # Tests against the analytical solution by Iskhakov et al (2017).
        "beta": 0.98,
        "next_wealth": {
            "wage": 20.0,
            "interest_rate": 0.0,
        },
        "utility": {
            "delta": 1.0,
        },
    },
    "low_delta": {
        # For very low values of delta we expect that most individuals work their entire
        # life.
        "beta": 0.98,
        "next_wealth": {
            "wage": 20.0,
            "interest_rate": 0.0,
        },
        "utility": {
            "delta": 0.1,
        },
    },
    "high_wage": {
        # For high wage we ...
        "beta": 0.98,
        "next_wealth": {
            "wage": 100.0,
            "interest_rate": 0.0,
        },
        "utility": {
            "delta": 0.1,
        },
    },
}


@pytest.mark.parametrize(("test_id", "params"), TEST_CASES.items())
def test_analytical_solution(test_id, params, test_model):
    """Test that the numerical solution matches the analytical solution.

    The analytical solution is from Iskhakov et al (2017) and is generated
    in the development repository: github.com/opensourceeconomics/lcm-dev.

    """
    with DATA.joinpath(f"{test_id}_v.pkl").open("rb") as file:
        analytical = pickle.load(file)  # noqa: S301

    if test_id == "low_delta":
        # nothing intersting happens in periods 4 and 5, so we skip them
        # to save runtime
        test_model["n_periods"] = 3

    # Prepare config parameters
    solve_model, params_template = get_lcm_function(model=test_model)

    params = tree_update(params_template, params)

    # Solve model using LCM
    vf_arr_list = solve_model(params=params)
    numerical_solution = np.stack(vf_arr_list)

    numerical = {
        "worker": numerical_solution[:, 0, :],
        "retired": numerical_solution[:, 1, :],
    }

    aaae(y=analytical["worker"], x=numerical["worker"], decimal=6)
    aaae(y=analytical["retired"], x=numerical["retired"], decimal=6)
