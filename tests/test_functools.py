from lcm.functools import all_as_kwargs


def test_all_as_kwargs():
    got = all_as_kwargs(
        args=(1, 2),
        kwargs={"c": 3},
        arg_names=["a", "b", "c"],
    )
    assert got == {"a": 1, "b": 2, "c": 3}


def test_all_as_kwargs_empty_args():
    got = all_as_kwargs(
        args=(),
        kwargs={"a": 1, "b": 2, "c": 3},
        arg_names=["a", "b", "c"],
    )
    assert got == {"a": 1, "b": 2, "c": 3}


def test_all_as_kwargs_empty_kwargs():
    got = all_as_kwargs(
        args=(1, 2, 3),
        kwargs={},
        arg_names=["a", "b", "c"],
    )
    assert got == {"a": 1, "b": 2, "c": 3}
