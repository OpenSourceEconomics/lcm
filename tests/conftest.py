from jax import config


def pytest_sessionstart(session):  # noqa: ARG001
    config.update("jax_enable_x64", val=True)
