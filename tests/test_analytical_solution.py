"""Testing against the analytical solution by Iskhakov et al (2017)."""
from pathlib import Path

import numpy as np
import pytest
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model
from numpy.testing import assert_array_almost_equal as aaae

TEST_DATA = Path(__file__).parent.resolve().joinpath("analytical_solution")


TEST_CASES = {
    "iskhakov_2017_five_periods": get_model("iskhakov_2017_five_periods"),
    "iskhakov_2017_low_delta": get_model("iskhakov_2017_low_delta"),
}


def mean_square_error(x, y, axis=None):
    return np.mean((x - y) ** 2, axis=axis)


@pytest.mark.parametrize(("model_name", "model_config"), TEST_CASES.items())
def test_analytical_solution(model_name, model_config):
    """Test that the numerical solution matches the analytical solution.

    The analytical solution is from Iskhakov et al (2017) and is generated
    in the development repository: github.com/opensourceeconomics/lcm-dev.

    """
    # Compute LCM solution
    # ==================================================================================
    solve_model, _ = get_lcm_function(model=model_config.model)

    vf_arr_list = solve_model(params=model_config.params)
    _numerical = np.stack(vf_arr_list)
    numerical = {
        "worker": _numerical[:, 0, :],
        "retired": _numerical[:, 1, :],
    }

    # Load analytical solution
    # ==================================================================================
    analytical = {
        _type: np.genfromtxt(
            TEST_DATA.joinpath(f"{model_name}__values_{_type}.csv"),
            delimiter=",",
        )
        for _type in ["worker", "retired"]
    }

    # Compare
    # ==================================================================================
    for _type in ["worker", "retired"]:
        _analytical = np.array(analytical[_type])
        _numerical = numerical[_type]

        # Compare the whole trajectory over time
        mse = mean_square_error(_analytical, _numerical, axis=0)
        # Exclude the first two initial wealth levels from the comparison, because the
        # numerical solution is unstable for very low wealth levels.
        aaae(mse[2:], 0, decimal=1)
