"""Testing against the analytical solution by Iskhakov et al (2017)."""
import pickle
from pathlib import Path

import numpy as np
import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON_NO_BORROWING
from numpy.testing import assert_array_almost_equal as aaae

# Path to analytical solution
DATA = Path(__file__).parent.resolve() / "analytical_solution"


def numerical_solution(params):
    """Numerical solution."""
    model = PHELPS_DEATON_NO_BORROWING
    model["n_periods"] = params["num_periods"]
    model["choices"]["consumption"]["start"] = 1
    model["choices"]["consumption"]["stop"] = 100
    model["choices"]["consumption"]["n_points"] = 10_000
    model["states"]["wealth"]["start"] = 1
    model["states"]["wealth"]["stop"] = 100
    model["states"]["wealth"]["n_points"] = 10_000

    solve_model, params_template = get_lcm_function(model=model)

    params_template["beta"] = params["beta"]
    params_template["next_wealth"]["wage"] = params["wage"]
    params_template["next_wealth"]["interest_rate"] = params["r"]
    params_template["utility"]["delta"] = params["delta"]

    numerical_solution = np.array(solve_model(params=params_template))

    return {
        "worker": numerical_solution[:, 0, :],
        "retired": numerical_solution[:, 1, :],
    }


# Define test cases
test_cases = {
    "iskhakov_2017": {
        "beta": 0.98,
        "delta": 1.0,
        "wage": float(20),
        "r": 0.0,
        "num_periods": 5,
    },
    "low_delta": {
        "beta": 0.98,
        "delta": 0.1,
        "wage": float(20),
        "r": 0.0,
        "num_periods": 3,
    },
    "high_wage": {
        "beta": 0.98,
        "delta": 1.0,
        "wage": float(100),
        "r": 0.0,
        "num_periods": 5,
    },
}


@pytest.mark.parametrize(("test_case", "params"), test_cases.items())
def test_analytical_solution(params, test_case):
    with open(DATA / f"{test_case}_v.pkl", "rb") as f:
        v_analytical = pickle.load(f)

    v_numerical = numerical_solution(params)

    aaae(y=v_analytical["worker"], x=v_numerical["worker"], decimal=6)
    aaae(y=v_analytical["retired"], x=v_numerical["retired"], decimal=6)
