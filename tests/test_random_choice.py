import jax
import jax.numpy as jnp
from lcm.random_choice import random_choice
from pybaum import tree_equal


def test_random_choice():
    key = jax.random.PRNGKey(0)
    probs = {"a": jnp.array([[0.0, 1], [1, 0]]), "b": jnp.array([[1.0, 0], [0, 1]])}
    labels = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}
    got = random_choice(key, probs=probs, labels=labels)
    assert tree_equal(got, {"a": jnp.array([2, 1]), "b": jnp.array([3, 4])})
