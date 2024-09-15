import re
from functools import partial

import jax.numpy as jnp
import pytest

from lcm import LinspaceGrid
from lcm.dispatchers import productmap
from lcm.function_representation import (
    _fail_if_interpolation_axes_are_not_last,
    _get_coordinate_finder,
    _get_interpolator,
    _get_label_translator,
    _get_lookup_function,
    get_function_representation,
)
from lcm.interfaces import (
    IndexerInfo,
    SpaceInfo,
)
from lcm.options import DefaultMapCoordinatesOptions


def test_function_evaluator_with_one_continuous_variable():
    wealth_grid = LinspaceGrid(start=-3, stop=3, n_points=7)

    space_info = SpaceInfo(
        axis_names=["wealth"],
        lookup_info={},
        interpolation_info={
            "wealth": wealth_grid,
        },
        indexer_infos=[],
    )

    vf_arr = jnp.pi * wealth_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_function_representation(
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        input_prefix="next_",
        interpolation_options=DefaultMapCoordinatesOptions,
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
        lookup_info={"working": [0, 1]},
        interpolation_info={},
        indexer_infos=[],
    )

    # create the evaluator
    evaluator = get_function_representation(
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        input_prefix="next_",
        interpolation_options=DefaultMapCoordinatesOptions,
    )

    # partial the function values into the evaluator
    func = partial(evaluator, vf_arr=vf_arr)

    # test the evaluator
    assert func(next_working=0) == 1
    assert func(next_working=1) == 2


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
        "retired": [0, 1],
        "working": [0, 1],
        "insured": [0, 1],
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
        "wealth": LinspaceGrid(start=100, stop=1100, n_points=6),
        "human_capital": LinspaceGrid(start=-3, stop=3, n_points=7),
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
    evaluator = get_function_representation(
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        interpolation_options=DefaultMapCoordinatesOptions,
    )

    # test the evaluator
    out = evaluator(
        retired=1,
        working=1,
        insured=0,
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
        "retired": [0, 1, 2],
        "working": [0, 1, 2],
        "insured": [0, 1],
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
        "wealth": LinspaceGrid(start=100, stop=1100, n_points=6),
        "human_capital": LinspaceGrid(start=-3, stop=3, n_points=7),
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
    evaluator = get_function_representation(
        space_info=space_info,
        name_of_values_on_grid="vf_arr",
        interpolation_options=DefaultMapCoordinatesOptions,
    )

    # test the evaluator
    out = evaluator(
        retired=0,
        working=1,
        insured=0,
        wealth=600,
        human_capital=1.5,
        state_indexer=indexer_array,
        vf_arr=vf_arr,
    )

    assert jnp.allclose(out, 601.5)


def test_get_label_translator_with_args():
    func = _get_label_translator(
        in_name="schooling",
    )
    assert func(1) == 1


def test_get_label_translator_with_kwargs():
    func = _get_label_translator(
        in_name="schooling",
    )
    assert func(schooling=1) == 1


def test_get_label_translator_wrong_kwarg():
    func = _get_label_translator(
        in_name="schooling",
    )
    with pytest.raises(
        TypeError,
        match=re.escape("translate_label() got unexpected keyword argument health"),
    ):
        func(health=1)


def test_get_lookup_function():
    indexer = jnp.arange(6).reshape(3, 2)
    func = _get_lookup_function(array_name="my_indexer", axis_names=["a", "b"])

    pure_lookup_func = partial(func, my_indexer=indexer)
    calculated = pure_lookup_func(a=2, b=0)
    assert calculated == 4


def test_get_coordinate_finder():
    find_coordinate = _get_coordinate_finder(
        in_name="wealth",
        grid=LinspaceGrid(start=0, stop=10, n_points=21),
    )

    calculated = find_coordinate(wealth=5.75)
    assert calculated == 11.5


def test_get_interpolator():
    interpolate = _get_interpolator(
        name_of_values_on_grid="vf",
        axis_names=["wealth", "working"],
        map_coordinates_options=DefaultMapCoordinatesOptions,
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


@pytest.mark.illustrative
def test_get_function_evaluator_illustrative():
    a_grid = LinspaceGrid(start=0, stop=1, n_points=3)

    space_info = SpaceInfo(
        axis_names=["a"],
        lookup_info={},
        interpolation_info={
            "a": a_grid,
        },
        indexer_infos=[],
    )

    values = jnp.pi * a_grid.to_jax() + 2

    # create the evaluator
    evaluator = get_function_representation(
        space_info=space_info,
        name_of_values_on_grid="values_name",
        input_prefix="prefix_",
        interpolation_options=DefaultMapCoordinatesOptions,
    )

    # partial the function values into the evaluator
    f = partial(evaluator, values_name=values)

    got = f(prefix_a=0.25)
    expected = jnp.pi * 0.25 + 2

    assert jnp.allclose(got, expected)


@pytest.mark.illustrative
def test_get_lookup_function_illustrative():
    values = jnp.array([0, 1, 4])
    func = _get_lookup_function(array_name="xyz", axis_names=["a"])
    pure_lookup_func = partial(func, xyz=values)

    assert pure_lookup_func(a=2) == 4


@pytest.mark.illustrative
def test_get_coordinate_finder_illustrative():
    find_coordinate = _get_coordinate_finder(
        in_name="a",
        grid=LinspaceGrid(start=0, stop=1, n_points=3),
    )

    assert find_coordinate(a=0) == 0
    assert find_coordinate(a=0.5) == 1
    assert find_coordinate(a=1) == 2
    assert find_coordinate(a=0.25) == 0.5


@pytest.mark.illustrative
def test_get_interpolator_illustrative():
    interpolate = _get_interpolator(
        name_of_values_on_grid="test_name",
        axis_names=["a", "b"],
        map_coordinates_options=DefaultMapCoordinatesOptions,
    )

    def f(a, b):
        return a - b

    prod_f = productmap(f, variables=["a", "b"])

    values = prod_f(a=jnp.arange(2, dtype=float), b=jnp.arange(3, dtype=float))

    assert interpolate(test_name=values, a=0.5, b=0) == 0.5
    assert interpolate(test_name=values, a=0.5, b=1) == -0.5
    assert interpolate(test_name=values, a=0, b=0.5) == -0.5
    assert interpolate(test_name=values, a=0.5, b=1.5) == -1


@pytest.mark.illustrative
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
