import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm.interfaces import StateActionSpace
from lcm.logging import get_logger
from lcm.max_continuous_actions import get_max_Q_over_c
from lcm.ndimage import map_coordinates
from lcm.solution.solve_brute import solve, solve_continuous_problem


def test_solve_brute():
    """Test solve brute with hand written inputs.

    Normally, these inputs would be created from a model specification. For now this can
    be seen as reference of what the functions that process a model specification need
    to produce.

    """
    # ==================================================================================
    # create the params
    # ==================================================================================
    params = {"beta": 0.9}

    # ==================================================================================
    # create the list of state_action_spaces
    # ==================================================================================
    _scs = StateActionSpace(
        discrete_actions={
            # pick [0, 1] such that no label translation is needed
            # lazy is like a type, it influences utility but is not affected by actions
            "lazy": jnp.array([0, 1]),
            "working": jnp.array([0, 1]),
        },
        continuous_actions={
            "consumption": jnp.array([0, 1, 2, 3]),
        },
        states={
            # pick [0, 1, 2] such that no coordinate mapping is needed
            "wealth": jnp.array([0.0, 1.0, 2.0]),
        },
        ordered_var_names=("lazy", "working", "wealth"),
    )
    state_action_spaces = {0: _scs, 1: _scs}

    # ==================================================================================
    # create the utility_and_feasibility functions
    # ==================================================================================

    def _utility_and_feasibility(consumption, lazy, wealth, working, vf_arr, params):
        _u = consumption - 0.2 * lazy * working
        _next_wealth = wealth + working - consumption
        _next_lazy = lazy
        _feasible = _next_wealth >= 0

        if vf_arr is None:
            cont_value = 0
        else:
            cont_value = _get_continuation_value(
                lazy=_next_lazy,
                wealth=_next_wealth,
                vf_arr=vf_arr,
            )

        beta = params["beta"]
        _utility = _u + beta * cont_value
        return _utility, _feasible

    def _get_continuation_value(lazy, wealth, vf_arr):
        continuous_part = vf_arr[lazy]
        return map_coordinates(
            input=continuous_part,
            coordinates=jnp.array([wealth]),
        )

    max_Q_over_c = get_max_Q_over_c(
        utility_and_feasibility=_utility_and_feasibility,
        continuous_action_variables=("consumption",),
    )

    max_Q_over_c_functions = {0: max_Q_over_c, 1: max_Q_over_c}

    # ==================================================================================
    # create max_Qc functions
    # ==================================================================================

    def max_Qc(Qc_values, params):  # noqa: ARG001
        """Take max over axis that corresponds to working."""
        return Qc_values.max(axis=1)

    max_Qc_functions = {0: max_Qc, 1: max_Qc}

    # ==================================================================================
    # call solve function
    # ==================================================================================

    solution = solve(
        params=params,
        state_action_spaces=state_action_spaces,
        max_Q_over_c_functions=max_Q_over_c_functions,
        max_Qc_functions=max_Qc_functions,
        logger=get_logger(debug_mode=False),
    )

    assert isinstance(solution, dict)


def test_solve_continuous_problem_no_vf_arr():
    state_action_space = StateActionSpace(
        discrete_actions={
            "a": jnp.array([0, 1.0]),
            "b": jnp.array([2, 3.0]),
            "c": jnp.array([4, 5, 6]),
        },
        continuous_actions={
            "d": jnp.arange(12.0),
        },
        states={},
        ordered_var_names=("a", "b", "c"),
    )

    def _utility_and_feasibility(a, c, b, d, vf_arr, params):  # noqa: ARG001
        util = d
        feasib = d <= a + b + c
        return util, feasib

    max_Q_over_c = get_max_Q_over_c(
        utility_and_feasibility=_utility_and_feasibility,
        continuous_action_variables=("d",),
    )

    expected = np.array([[[6.0, 7, 8], [7, 8, 9]], [[7, 8, 9], [8, 9, 10]]])

    got = solve_continuous_problem(
        state_action_space,
        max_Q_over_c,
        vf_arr=None,
        params={},
    )
    aaae(got, expected)
