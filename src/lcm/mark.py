"""Collection of LCM marking decorators."""
import functools
from typing import NamedTuple


class StochasticOptions(NamedTuple):
    """Options passed to the stochastic decorator."""


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
    stochastic_options = StochasticOptions(*args, **kwargs)

    def decorator_stochastic(func):
        @functools.wraps(func)
        def wrapper_mark_stochastic(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper_mark_stochastic.is_stochastic = True
        wrapper_mark_stochastic.stochastic_options = stochastic_options
        return wrapper_mark_stochastic

    return decorator_stochastic(func) if callable(func) else decorator_stochastic
