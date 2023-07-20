import jax

from lcm import mark

jax.config.update("jax_platform_name", "cpu")


__all__ = ["mark"]
