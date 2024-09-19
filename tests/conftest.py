from dataclasses import make_dataclass

import pytest
from jax import config


def pytest_sessionstart(session):  # noqa: ARG001
    config.update("jax_enable_x64", val=True)


def _category_class_factory(values: list[int]):
    init = [(f"cat{i}", int, value) for i, value in enumerate(values)]
    return make_dataclass("CategoryClass", init)


@pytest.fixture(scope="session")
def category_class_factory():
    return _category_class_factory


@pytest.fixture(scope="session")
def binary_category_class(category_class_factory):
    return category_class_factory([0, 1])
