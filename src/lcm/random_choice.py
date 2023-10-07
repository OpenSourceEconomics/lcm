from functools import partial

import jax


def random_choice(key, probs, labels):
    """Draw multiple random choices.

    Args:
        key (jax.random.PRNGKey): Random keys. One for each simulation unit.
        probs (jax.numpy.array): 2d array of probabilities. Second dimension must be
            the same length as the first dimension of labels.
        labels (jax.numpy.array): 1d array of labels.

    Returns:
        jax.numpy.array: The index

    """
    keys = jax.random.split(key, probs.shape[0])
    return _vmapped_random_choice(keys, probs, labels)


@partial(jax.vmap, in_axes=(0, 0, None))
def _vmapped_random_choice(key, probs, labels):
    return jax.random.choice(key, a=labels, p=probs)
