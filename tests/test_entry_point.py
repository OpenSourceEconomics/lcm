from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON


def test_get_lcm_function_with_solve_target():
    get_lcm_function(model=PHELPS_DEATON)
