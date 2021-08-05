import inspect

from lcm.dag import concatenate_functions


def _utility(_consumption, _leisure):
    return _consumption + _leisure


def _leisure(working):
    return 24 - working


def _consumption(working, wage):
    return wage * working


def _unrelated(working):  # noqa: U100
    raise NotImplementedError()


def test_concatenate_functions():
    concatenated = concatenate_functions(
        functions=[_utility, _unrelated, _leisure, _consumption],
        target="_utility",
    )

    calculated_res = concatenated(wage=5, working=8)
    expected_res = 56
    assert calculated_res == expected_res

    calculated_args = set(inspect.signature(concatenated).parameters)
    expected_args = {"wage", "working"}

    assert calculated_args == expected_args
