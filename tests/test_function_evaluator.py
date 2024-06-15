from functools import partial

import jax.numpy as jnp
import pytest
from lcm.dispatchers import productmap
from lcm.function_evaluator import (
    _fail_if_interpolation_axes_are_not_last,
    _get_coordinate_finder,
    _get_interpolator,
    _get_lookup_function,
    get_function_evaluator,
    get_label_translator,
)
from lcm.grids import linspace
from lcm.interfaces import (
    ContinuousGridInfo,
    ContinuousGridSpec,
    IndexerInfo,
    SpaceInfo,
)


def test_function_evaluator_with_one_continuous_variable():
    grid_info = ContinuousGridInfo(start=-3, stop=3, n_points=7)

    space_info = SpaceInfo(
        axis_names=["wealth"],
        lookup_info={},
        interpolation_info={
            "wealth": ContinuousGridSpec(kind="linspace", info=grid_info),
        },
        indexer_infos=[],
    )

    grid = linspace(**grid_info._asdict())
    vf_arr = jnp.pi * grid + 2

    # create the evaluator
    evaluator = get_function_evaluator(
        space_info=space_info,
        data_name="vf_arr",
        input_prefix="next_",
    )

    # partial the function values into the evaluator
    func = partial(evaluator, vf_arr=vf_arr)

    # test the evaluator
    got = func(next_wealth=0.5)
    expected = 0.5 * jnp.pi + 2
    assert jnp.allclose(got, expected)


def test_function_evaluator_with_one_discrete_variable():
    vf_arr = jnp.array([1, 2])

    space_info = SpaceInfo(
        axis_names=["working"],
        lookup_info={"working": [True, False]},
        interpolation_info={},
        indexer_infos=[],
    )

    # create the evaluator
    evaluator = get_function_evaluator(
        space_info=space_info,
        data_name="vf_arr",
        input_prefix="next_",
    )

    # partial the function values into the evaluator
    func = partial(evaluator, vf_arr=vf_arr)

    # test the evaluator
    assert func(next_working=True) == 1
    assert func(next_working=False) == 2


def test_function_evaluator():
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
    lookup_info = {
        "retired": [True, False],
        "working": [0, 1],
        "insured": ["yes", "no"],
    }

    # create an indexer for the sparse discrete part
    indexer_infos = [
        IndexerInfo(
            axis_names=["retired", "working"],
            name="state_indexer",
            out_name="state_index",
        ),
    ]

    indexer_array = jnp.array([[-1, 0], [1, 2]])

    # create info on continuous grids
    interpolation_info = {
        "wealth": ContinuousGridSpec(
            kind="linspace",
            info=ContinuousGridInfo(start=100, stop=1100, n_points=6),
        ),
        "human_capital": ContinuousGridSpec(
            kind="linspace",
            info=ContinuousGridInfo(start=-3, stop=3, n_points=7),
        ),
    }

    # create info on axis of value function array
    axis_names = ["state_index", "insured", "wealth", "human_capital"]

    space_info = SpaceInfo(
        axis_names=axis_names,
        lookup_info=lookup_info,
        interpolation_info=interpolation_info,
        indexer_infos=indexer_infos,
    )

    # create the evaluator
    evaluator = get_function_evaluator(
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
        state_indexer=indexer_array,
        vf_arr=vf_arr,
    )

    assert jnp.allclose(out, 1001.5)


def test_function_evaluator_longer_indexer():
    """Test get_precalculated_function_evaluator in an extended example.

    - One sparse discrete state variable: retired ('working', 'part retired', retired)
    - One sparse discrete choice variable: working (0, 1, 2)
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
    discrete_part = jnp.arange(10).repeat(6 * 7).reshape((5, 2, 6, 7)) * 100
    cont_func = productmap(lambda x, y: x + y, ["x", "y"])
    cont_part = cont_func(x=jnp.linspace(100, 1100, 6), y=jnp.linspace(-3, 3, 7))
    vf_arr = discrete_part + cont_part

    # create info on discrete variables
    lookup_info = {
        "retired": ["working", "part-retired", "retired"],
        "working": [0, 1, 2],
        "insured": ["yes", "no"],
    }

    # create an indexer for the sparse discrete part
    indexer_infos = [
        IndexerInfo(
            axis_names=["retired", "working"],
            name="state_indexer",
            out_name="state_index",
        ),
    ]

    indexer_array = jnp.array([[-1, 0, 1], [2, 3, -1], [4, -1, -1]])

    # create info on continuous grids
    interpolation_info = {
        "wealth": ContinuousGridSpec(
            kind="linspace",
            info=ContinuousGridInfo(start=100, stop=1100, n_points=6),
        ),
        "human_capital": ContinuousGridSpec(
            kind="linspace",
            info=ContinuousGridInfo(start=-3, stop=3, n_points=7),
        ),
    }

    # create info on axis of value function array
    axis_names = ["state_index", "insured", "wealth", "human_capital"]

    space_info = SpaceInfo(
        axis_names=axis_names,
        lookup_info=lookup_info,
        interpolation_info=interpolation_info,
        indexer_infos=indexer_infos,
    )

    # create the evaluator
    evaluator = get_function_evaluator(
        space_info=space_info,
        data_name="vf_arr",
    )

    # test the evaluator
    out = evaluator(
        retired="working",
        working=1,
        insured="yes",
        wealth=600,
        human_capital=1.5,
        state_indexer=indexer_array,
        vf_arr=vf_arr,
    )

    assert jnp.allclose(out, 601.5)


def test_get_label_translator():
    grid = jnp.array([9, 10, 13])

    func = get_label_translator(
        labels=grid,
        in_name="schooling",
    )

    assert func(schooling=10) == 1


def test_get_lookup_function():
    indexer = jnp.arange(6).reshape(3, 2)
    func = _get_lookup_function(array_name="my_indexer", axis_names=["a", "b"])

    pure_lookup_func = partial(func, my_indexer=indexer)
    calculated = pure_lookup_func(a=2, b=0)
    assert calculated == 4


def test_get_coordinate_finder():
    find_coordinate = _get_coordinate_finder(
        in_name="wealth",
        grid_type="linspace",
        grid_info=ContinuousGridInfo(start=0, stop=10, n_points=21),
    )

    calculated = find_coordinate(wealth=5.75)
    assert calculated == 11.5


def test_get_interpolator():
    interpolate = _get_interpolator(
        data_name="vf",
        axis_names=["wealth", "working"],
        map_coordinates_options=None,
    )

    def _utility(wealth, working):
        return 2 * wealth - working

    prod_utility = productmap(_utility, variables=["wealth", "working"])

    values = prod_utility(
        wealth=jnp.arange(4, dtype=float),
        working=jnp.arange(3, dtype=float),
    )

    calculated = interpolate(vf=values, wealth=2.5, working=2)

    assert calculated == 3


# ======================================================================================
# Illustrative
# ======================================================================================


@pytest.mark.illustrative()
def test_get_function_evaluator_illustrative():
    grid_info = ContinuousGridInfo(start=0, stop=1, n_points=3)

    space_info = SpaceInfo(
        axis_names=["a"],
        lookup_info={},
        interpolation_info={
            "a": ContinuousGridSpec(kind="linspace", info=grid_info),
        },
        indexer_infos=[],
    )

    grid = linspace(**grid_info._asdict())

    values = jnp.pi * grid + 2

    # create the evaluator
    evaluator = get_function_evaluator(
        space_info=space_info,
        data_name="values_name",
        input_prefix="prefix_",
    )

    # partial the function values into the evaluator
    f = partial(evaluator, values_name=values)

    got = f(prefix_a=0.25)
    expected = jnp.pi * 0.25 + 2

    assert jnp.allclose(got, expected)


@pytest.mark.illustrative()
def test_get_label_translator_illustrative():
    # Range (fast "lookup")
    # ==================================================================================
    f = get_label_translator(
        labels=jnp.array([0, 1, 2, 3]),
        in_name="a",
    )
    assert f(a=1) == 1

    # Non-range (slow lookup compared to range)
    # ==================================================================================
    g = get_label_translator(
        labels=jnp.array([-1, 0, 2, 10]),
        in_name="a",
    )
    assert g(a=0) == 1
    assert g(a=10) == 3


@pytest.mark.illustrative()
def test_get_lookup_function_illustrative():
    values = jnp.array([0, 1, 4])
    func = _get_lookup_function(array_name="xyz", axis_names=["a"])
    pure_lookup_func = partial(func, xyz=values)

    assert pure_lookup_func(a=2) == 4


@pytest.mark.illustrative()
def test_get_coordinate_finder_illustrative():
    find_coordinate = _get_coordinate_finder(
        in_name="a",
        grid_type="linspace",
        grid_info=ContinuousGridInfo(start=0, stop=1, n_points=3),
    )

    assert find_coordinate(a=0) == 0
    assert find_coordinate(a=0.5) == 1
    assert find_coordinate(a=1) == 2
    assert find_coordinate(a=0.25) == 0.5


@pytest.mark.illustrative()
def test_get_interpolator_illustrative():
    interpolate = _get_interpolator(
        data_name="data_name",
        axis_names=["a", "b"],
        map_coordinates_options=None,
    )

    def f(a, b):
        return a - b

    prod_f = productmap(f, variables=["a", "b"])

    values = prod_f(a=jnp.arange(2, dtype=float), b=jnp.arange(3, dtype=float))

    assert interpolate(data_name=values, a=0.5, b=0) == 0.5
    assert interpolate(data_name=values, a=0.5, b=1) == -0.5
    assert interpolate(data_name=values, a=0, b=0.5) == -0.5
    assert interpolate(data_name=values, a=0.5, b=1.5) == -1


@pytest.mark.illustrative()
def test_fail_if_interpolation_axes_are_not_last_illustrative():
    # Empty intersection of axis_names and interpolation_info
    # ==================================================================================

    space_info = SpaceInfo(
        axis_names=["a", "b"],
        interpolation_info={
            "c": None,
        },
        lookup_info=None,
        indexer_infos=None,
    )

    _fail_if_interpolation_axes_are_not_last(space_info)  # does not fail

    # Non-empty intersection but correct order
    # ==================================================================================

    space_info = SpaceInfo(
        axis_names=["a", "b", "c"],
        interpolation_info={
            "b": None,
            "c": None,
            "d": None,
        },
        lookup_info=None,
        indexer_infos=None,
    )

    _fail_if_interpolation_axes_are_not_last(space_info)  # does not fail

    # Non-empty intersection and in-correct order
    # ==================================================================================

    space_info = SpaceInfo(
        axis_names=["b", "c", "a"],  # "b", "c" are not last anymore
        interpolation_info={
            "b": None,
            "c": None,
            "d": None,
        },
        lookup_info=None,
        indexer_infos=None,
    )

    with pytest.raises(ValueError, match="Interpolation axes need to be the last"):
        _fail_if_interpolation_axes_are_not_last(space_info)
