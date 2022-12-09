import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON
from lcm.example_models import PHELPS_DEATON_WITH_FILTERS
from lcm.example_models import PHELPS_DEATON_WITH_SHOCKS

MODELS = {
    "simple": PHELPS_DEATON,
    "with_shocks": PHELPS_DEATON_WITH_SHOCKS,
    "with_filters": PHELPS_DEATON_WITH_FILTERS,
}


@pytest.mark.parametrize("model", list(MODELS.values()), ids=list(MODELS))
def test_get_lcm_function_with_solve_target(model):
    get_lcm_function(model=model)
