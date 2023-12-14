from collections.abc import Callable
from typing import Any

from jax.typing import ArrayLike

from lcm.interfaces import Space


def backward_induction(
    params: Any,
    solve_continuous_problem: list[Callable],
    solve_discrete_problem: list[Callable],
    continuous_choice_grids: list[dict[str, ArrayLike]],
    state_choice_spaces: list[Space],
    state_indexers: list[ArrayLike],
) -> list[ArrayLike]:
    """Solve a model using backward induction.

    All list dimensions correspond to the model periods.

    Args:
        params: Dict of model parameters.

        solve_continuous_problem: List of functions needed to solve the agent's
            continuous choice problem, conditional on the remaining discrete choices.
            One function for each model period.

        solve_discrete_problem: List of functions that take the conditional continuation
            value and calculates the maximum over the remaining discrete choices. One
            function for each model period.

        continuous_choice_grids: List of dicts with 1d grids for each continuous choice
            variable.

        state_choice_spaces: List with one state_choice_space per period.

        state_indexers: List of dicts with length n_periods. Each dict contains one or
            several state indexers.

    Returns:
        List with the corresponding value function array for each period.

    """
    # Setup
    # ==================================================================================
    n_periods = len(state_choice_spaces)
    reversed_solution = []
    # vf_arr represents the value function of next period, which will be updated in each
    # iteration of the backward induction loop.
    vf_arr = None

    # Backwards induction loop
    # ==================================================================================
    for period in reversed(range(n_periods)):
        # Solve the continuous problem, conditional on discrete choices
        conditional_continuation_values = solve_continuous_problem[period](
            **state_choice_spaces[period].dense_vars,
            **continuous_choice_grids[period],
            **state_choice_spaces[period].sparse_vars,
            **state_indexers[period],
            vf_arr=vf_arr,
            params=params,
        )

        # Solve the discrete problem, given the conditional continuation values
        vf_arr = solve_discrete_problem[period](conditional_continuation_values)
        reversed_solution.append(vf_arr)

    return list(reversed(reversed_solution))
