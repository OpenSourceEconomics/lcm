from functools import partial

import jax.numpy as jnp
from lcm.evaluate_precalculated_function import get_discrete_grid_position_finder
from lcm.evaluate_precalculated_function import get_indexer_wrapper


def test_get_indexer_wrapper():
    indexer = jnp.arange(6).reshape(3, 2)
    func = get_indexer_wrapper(
        indexer_name="my_indexer", axis_order=["a", "b"], out_name="state_pos"
    )

    pure_lookup_func = partial(func, my_indexer=indexer)
    calculated = pure_lookup_func(a=2, b=0, my_indexer=indexer)
    assert calculated == 4
    assert func.__name__ == "state_pos"


def test_get_discrete_grid_position_finder():
    grid = jnp.array([9, 10, 13])

    func = get_discrete_grid_position_finder(
        grid=grid,
        in_name="schooling",
    )

    assert func(schooling=10) == 1
    assert func.__name__ == "schooling_pos"
