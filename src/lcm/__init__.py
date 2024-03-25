import jax

from lcm import mark


jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_transfer_guard", "allow")

__all__ = ["mark"]
