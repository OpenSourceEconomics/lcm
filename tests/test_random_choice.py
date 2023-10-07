import jax
import jax.numpy as jnp
from lcm.random_choice import random_choice


def test_random_choice():
    key = jax.random.PRNGKey(0)
    probs = jnp.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    labels = jnp.array([1, 2, 3])
    got = random_choice(key, probs=probs, labels=labels)
    assert jnp.array_equal(got, jnp.array([3, 1, 2]))
