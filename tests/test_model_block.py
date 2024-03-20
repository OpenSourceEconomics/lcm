from lcm.model_block import ModelBlock

from tests.test_models.deterministic import get_model_config


def test_model_block():
    model_config = get_model_config("iskhakov_et_al_2017", 3)
    model_block = ModelBlock(model_config)
    breakpoint()
