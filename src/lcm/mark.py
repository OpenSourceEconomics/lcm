"""Collection of LCM marking decorators."""

import functools
from dataclasses import dataclass


@dataclass(frozen=True)
class StochasticInfo:
    """Information on the stochastic nature of user provided functions."""


def stochastic(
    func,
    *args,
    **kwargs,
):
    """Decorator to mark a function as stochastic and add information.

    Args:
        func (callable): The function to be decorated.
        *args (list): Positional arguments to be passed to the StochasticInfo.
        **kwargs (dict): Keyword arguments to be passed to the StochasticInfo.

    Returns:
        callable: The decorated function

    """
    stochastic_info = StochasticInfo(*args, **kwargs)

    def decorator_stochastic(func):
        @functools.wraps(func)
        def wrapper_mark_stochastic(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper_mark_stochastic._stochastic_info = stochastic_info
        return wrapper_mark_stochastic

    return decorator_stochastic(func) if callable(func) else decorator_stochastic
