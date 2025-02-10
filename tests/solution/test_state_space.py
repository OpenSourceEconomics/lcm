from lcm.input_processing import process_model
from lcm.solution.state_space import (
    create_state_choice_space,
)
from tests.test_models import get_model_config


def test_create_state_choice_space():
    _model = process_model(
        get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3),
    )
    create_state_choice_space(
        model=_model,
        is_last_period=False,
    )
