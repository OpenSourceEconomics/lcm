import functools
from typing import NamedTuple


class StochasticInfo(NamedTuple):
    pass


def stochastic(
    func,
    *args,
    **kwargs,
):
    """Decorator to mark a function as stochastic and add information.

    Args:
        func (callable): The function to be decorated

    Returns:
        callable: The decorated function

    """
    stochastic_info = StochasticInfo(*args, **kwargs)

    def decorator_stochastic(func):
        @functools.wraps(func)
        def wrapper_mark_minimizer(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper_mark_minimizer._stochastic_info = stochastic_info
        return wrapper_mark_minimizer

    if callable(func):
        return decorator_stochastic(func)

    else:
        return decorator_stochastic
