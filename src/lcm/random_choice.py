from functools import partial

import jax


def random_choice(
    key: jax.Array,
    probs: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Draw multiple random choices.

    Args:
        key: Random key.
        probs: 2d array of probabilities. Second dimension must be
            the same length as the first dimension of labels.
        labels: 1d array of labels.

    Returns:
        Selected labels. 1d array of length len(probs).

    """
    keys = jax.random.split(key, probs.shape[0])
    return _vmapped_random_choice(keys, probs, labels)


@partial(jax.vmap, in_axes=(0, 0, None))
def _vmapped_random_choice(
    key: jax.Array, probs: jax.Array, labels: jax.Array
) -> jax.Array:
    return jax.random.choice(key, a=labels, p=probs)
