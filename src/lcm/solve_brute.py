import jax.numpy as jnp
from lcm.dispatchers import gridmap


def contsolve_last_period(
    state_choice_space,
    utility_and_feasibility,
    continuous_choice_grids,
):
    """Solve the agent's continuous choices problem problem in the last period.

    Args:
        state_choice_space
        utility_and_feasibility (callable): Function that returns a tuple where the
            first entry is the utility and the second is a bool that indicates
            feasibility. The function only depends on states, discret and continuous
            choices. Parameters are already partialled in.

    Returns:
        np.ndarray: Numpy array with continuation values for each state. The number
            and order of dimensions is defined by the ``gridmap`` function.

    """
    n_simple = len(state_choice_space["simple_variables"])
    n_cont_choices = len(continuous_choice_grids)

    max_axes = tuple(range(n_simple, n_simple + n_cont_choices))

    gridmapped = gridmap(
        func=utility_and_feasibility,
        simple_variables=list(state_choice_space["simple_variables"])
        + list(continuous_choice_grids),
        complex_variables=list(state_choice_space["complex_variables"]),
    )

    utilities, feasibilities = gridmapped(
        **state_choice_space["simple_variables"],
        **continuous_choice_grids,
        **state_choice_space["complex_variables"],
    )

    best = utilities.max(axis=max_axes, where=feasibilities, initial=-jnp.inf)

    return best
