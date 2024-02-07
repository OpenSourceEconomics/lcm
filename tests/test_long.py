import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models.example_models_long import PARAMS, PHELPS_DEATON

SKIP_REASON = """The test is designed to run approximately 1 minute on a standard
laptop, such that we can differentiate the performance of running LCM on a GPU versus
on the CPU."""


@pytest.skip(reason=SKIP_REASON)
def test_long():
    solve_model, template = get_lcm_function(PHELPS_DEATON, targets="solve")
    solve_model(PARAMS)
