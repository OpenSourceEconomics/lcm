"""Testing against the analytical solution by Iskhakov et al (2017)."""
import pickle
from pathlib import Path

import numpy as np
import pytest
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model
from numpy.testing import assert_array_almost_equal as aaae

DATA = Path(__file__).parent.resolve().joinpath("analytical_solution")


TEST_CASES = {
    "iskhakov_2017_five_periods": get_model("iskhakov_2017_five_periods"),
    "iskhakov_2017_low_delta": get_model("iskhakov_2017_low_delta"),
}


@pytest.mark.parametrize(("test_id", "model"), TEST_CASES.items())
def test_analytical_solution_values(test_id, model):
    """Test that the numerical solution matches the analytical solution.

    The analytical solution is from Iskhakov et al (2017) and is generated
    in the development repository: github.com/opensourceeconomics/lcm-dev.

    """
    with DATA.joinpath(f"{test_id}_v.pkl").open("rb") as file:
        analytical = pickle.load(file)  # noqa: S301

    # Prepare config parameters
    solve_model, _ = get_lcm_function(model=model.model)

    # Solve model using LCM
    vf_arr_list = solve_model(params=model.params)
    numerical_solution = np.stack(vf_arr_list)

    numerical = {
        "worker": numerical_solution[:, 0, :],
        "retired": numerical_solution[:, 1, :],
    }

    aaae(y=analytical["worker"], x=numerical["worker"], decimal=6)
    aaae(y=analytical["retired"], x=numerical["retired"], decimal=6)
