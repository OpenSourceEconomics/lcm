"""Testing against the analytical solution by Iskhakov et al (2017)."""
import jax
import numpy as np
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model

TEST_CASES = {
    "iskhakov_2017_five_periods": get_model("iskhakov_2017_five_periods"),
    "iskhakov_2017_low_delta": get_model("iskhakov_2017_low_delta"),
}


def mean_square_error(x, y, axis=None):
    return np.mean((x - y) ** 2, axis=axis)


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


with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    test_analytical_solution(
        "iskhakov_2017_five_periods",
        get_model("iskhakov_2017_five_periods"),
    )
