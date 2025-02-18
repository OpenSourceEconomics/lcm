from collections import Counter
from collections.abc import Iterable
from itertools import chain
from typing import Any, Self, TypeVar

T = TypeVar("T")


def find_duplicates(*containers: Iterable[T]) -> set[T]:
    combined = chain.from_iterable(containers)
    counts = Counter(combined)
    return {v for v, count in counts.items() if count > 1}


class ReplaceMixin:
    """Mixin for replacing init-attributes of a class."""

    def replace(self: Self, **kwargs: Any) -> Self:  # noqa: ANN401
        """Replace the init-attributes of the class with the given values."""
        return type(self)(**{**self.__dict__, **kwargs})
