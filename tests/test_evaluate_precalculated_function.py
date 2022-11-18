from functools import partial

import jax.numpy as jnp
from lcm.create_state_space import Grid
from lcm.create_state_space import Indexer
from lcm.create_state_space import SpaceInfo
from lcm.dispatchers import productmap
from lcm.evaluate_precalculated_function import get_coordinate_finder
from lcm.evaluate_precalculated_function import get_interpolator
from lcm.evaluate_precalculated_function import get_label_translator
from lcm.evaluate_precalculated_function import get_lookup_function
from lcm.evaluate_precalculated_function import get_precalculated_function_evaluator


def test_get_precalculated_function_evaluator():
    """Test get_precalculated_function_evaluator in simple example.

    - One sparse discrete state variable: retired (True, False)
    - One sparse discrete choice variable: working (True, False)
    - One dense discrete choice variable: insured ("yes", "no")
    - Two dense continuous state variables:
        - wealth (linspace(100, 1100, 6))
        - human_capital (linspace(-3, 3, 7))

    The utility function is wealth + human_capital + c. c takes a different
    value for each discrete state choice combination.

    The setup of space_info here is quite long. Usually these inputs will be generated
    from a model specification.

    """
    # create a value function array
    discrete_part = jnp.arange(6).repeat(6 * 7).reshape((3, 2, 6, 7)) * 100
    cont_func = productmap(lambda x, y: x + y, ["x", "y"])
    cont_part = cont_func(x=jnp.linspace(100, 1100, 6), y=jnp.linspace(-3, 3, 7))
    vf_arr = discrete_part + cont_part

    # create info on discrete variables
    lookup_axes = {
        "retired": [True, False],
        "working": [0, 1],
        "insured": ["yes", "no"],
    }

    # create an indexer for the sparse discrete part
    indexers = [
        Indexer(
            axis_order=["retired", "working"],
            name="state_indexer",
            out_name="state_index",
            indexer=jnp.array([[-1, 0], [1, 2]]),
        )
    ]

    # create info on continuous grids
    interpolation_axes = {
        "wealth": Grid(
            kind="linspace",
            specs={"start": 100, "stop": 1100, "n_points": 6},
        ),
        "human_capital": Grid(
            kind="linspace", specs={"start": -3, "stop": 3, "n_points": 7}
        ),
    }

    # create info on axis of value function array
    axis_order = ["state_index", "insured", "wealth", "human_capital"]

    space_info = SpaceInfo(
        axis_order=axis_order,
        lookup_axes=lookup_axes,
        interpolation_axes=interpolation_axes,
        indexers=indexers,
    )

    # create the evaluator
    evaluator = get_precalculated_function_evaluator(
        space_info=space_info,
        data_name="vf_arr",
    )

    # test the evaluator
    out = evaluator(
        retired=False,
        working=1,
        insured="yes",
        wealth=600,
        human_capital=1.5,
        state_indexer=indexers[0].indexer,
        vf_arr=vf_arr,
    )

    assert jnp.allclose(out, 1001.5)


def test_get_lookup_function():
    indexer = jnp.arange(6).reshape(3, 2)
    func = get_lookup_function(
        array_name="my_indexer", axis_order=["a", "b"], out_name="state_pos"
    )

    pure_lookup_func = partial(func, my_indexer=indexer)
    calculated = pure_lookup_func(a=2, b=0)
    assert calculated == 4
    assert func.__name__ == "state_pos"


def test_get_label_translator():
    grid = jnp.array([9, 10, 13])

    func = get_label_translator(
        labels=grid,
        in_name="schooling",
    )

    assert func(schooling=10) == 1
    assert func.__name__ == "schooling_pos"


def test_get_coordinate_finder():

    find_coordinate = get_coordinate_finder(
        in_name="wealth",
        grid_type="linspace",
        grid_info={"start": 0, "stop": 10, "n_points": 21},
    )

    calculated = find_coordinate(wealth=5.75)
    assert calculated == 11.5


def test_get_interpolator():

    interpolate = get_interpolator(value_name="vf", axis_order=["wealth", "working"])

    def _utility(wealth, working):
        return 2 * wealth - working

    prod_utility = productmap(_utility, variables=["wealth", "working"])

    values = prod_utility(wealth=jnp.arange(4), working=jnp.arange(3))

    calculated = interpolate(vf=values, wealth=2.5, working=2)

    assert calculated == 3
