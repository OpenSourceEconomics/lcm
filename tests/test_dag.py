import inspect

from lcm.dag import concatenate_functions
from lcm.dag import get_ancestors


def _utility(_consumption, _leisure):
    return _consumption + _leisure


def _leisure(working):
    return 24 - working


def _consumption(working, wage):
    return wage * working


def _unrelated(working):  # noqa: U100
    raise NotImplementedError()


def _complete_utility(wage, working):
    """The function that we try to generate dynamically."""
    leis = _leisure(working)
    cons = _consumption(working, wage)
    util = leis + cons
    return util


def test_concatenate_functions_single_target():
    concatenated = concatenate_functions(
        functions=[_utility, _unrelated, _leisure, _consumption],
        targets="_utility",
    )

    calculated_res = concatenated(wage=5, working=8)

    expected_res = _complete_utility(wage=5, working=8)
    assert calculated_res == expected_res

    calculated_args = set(inspect.signature(concatenated).parameters)
    expected_args = {"wage", "working"}

    assert calculated_args == expected_args


def test_concatenate_functions_multi_target():
    concatenated = concatenate_functions(
        functions=[_utility, _unrelated, _leisure, _consumption],
        targets=["_utility", "_consumption"],
    )

    calculated_res = concatenated(wage=5, working=8)

    expected_res = (
        _complete_utility(wage=5, working=8),
        _consumption(wage=5, working=8),
    )
    assert calculated_res == expected_res

    calculated_args = set(inspect.signature(concatenated).parameters)
    expected_args = {"wage", "working"}

    assert calculated_args == expected_args


def test_get_ancestors_many_ancestors():
    calculated = get_ancestors(
        functions=[_utility, _unrelated, _leisure, _consumption],
        targets="_utility",
    )
    expected = {"_consumption", "_leisure", "working", "wage"}

    assert calculated == expected


def test_get_ancestors_few_ancestors():
    calculated = get_ancestors(
        functions=[_utility, _unrelated, _leisure, _consumption],
        targets="_unrelated",
    )

    expected = {"working"}

    assert calculated == expected


def test_get_ancestors_multiple_targets():
    calculated = get_ancestors(
        functions=[_utility, _unrelated, _leisure, _consumption],
        targets=["_unrelated", "_consumption"],
    )

    expected = {"wage", "working"}
    assert calculated == expected
