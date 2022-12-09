from lcm.example_models import PHELPS_DEATON_WITH_FILTERS
from lcm.process_model import process_model


def test_process_model_with_filters_runs():
    process_model(PHELPS_DEATON_WITH_FILTERS)
