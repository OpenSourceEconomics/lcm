from functools import partial

import jax


def random_choice(key, probs, labels):
    """Draw random choice from n-dimensional array.

    Args:
        key (jax.random.PRNGKey): Random key.
        probs (dict): Dictionary of probabilities. Probabilities are assumed to be
            2d arrays with the first dimension corresponding simulation units.
        labels (dict): Dictionary of labels. Labels are assumed to be 1d arrays.

    Returns:
        dict: Choice of label, for each simulation unit.

    """
    var_names = list(probs)
    keys = jax.random.split(key, len(var_names))
    return {
        name: _random_choice(key, probs[name], labels[name])
        for name, key in zip(var_names, keys, strict=True)
    }


def _random_choice(key, probs, labels):
    """Draw multiple random choices.

    Args:
        key (jax.random.PRNGKey): Random key.
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
