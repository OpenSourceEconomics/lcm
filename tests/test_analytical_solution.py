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


def numerical_solution(input_params):
    """Numerical solution."""
    config_update = {
        "n_periods": input_params["n_periods"],
        "choices": {
            "consumption": {
                "grid_type": "linspace",
                "start": 1,
                "stop": 100,
                "n_points": 10_000,
            },
            "retirement": {
                "options": [0, 1],
            },
        },
        "states": {
            "lagged_retirement": {
                "options": [0, 1],
            },
            "wealth": {
                "grid_type": "linspace",
                "start": 1,
                "stop": 100,
                "n_points": 10_000,
            },
        },
    }
    model = {**PHELPS_DEATON_NO_BORROWING, **config_update}
    solve_model, params_template = get_lcm_function(model=model)

    model_params_update = {
        "beta": input_params["beta"],
        "next_wealth": {
            "wage": input_params["wage"],
            "interest_rate": input_params["r"],
        },
        "utility": {
            "delta": input_params["delta"],
        },
    }

    model_params = {**params_template, **model_params_update}

    numerical_solution = np.array(solve_model(params=model_params))

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
        "n_periods": 5,
    },
    "low_delta": {
        "beta": 0.98,
        "delta": 0.1,
        "wage": float(20),
        "r": 0.0,
        "n_periods": 3,
    },
    "high_wage": {
        "beta": 0.98,
        "delta": 1.0,
        "wage": float(100),
        "r": 0.0,
        "n_periods": 5,
    },
}


@pytest.mark.parametrize(("test_case", "params"), test_cases.items())
def test_analytical_solution(params, test_case):
    with Path.open(DATA / f"{test_case}_v.pkl", "rb") as f:
        v_analytical = pickle.load(f)  # noqa: S301

    v_numerical = numerical_solution(params)

    aaae(y=v_analytical["worker"], x=v_numerical["worker"], decimal=6)
    aaae(y=v_analytical["retired"], x=v_numerical["retired"], decimal=6)
