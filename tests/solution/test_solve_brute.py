import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm.interfaces import StateActionSpace
from lcm.logging import get_logger
from lcm.max_continuous_actions import get_max_Q_over_c
from lcm.ndimage import map_coordinates
from lcm.solution.solve_brute import solve


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
        states_and_discrete_actions_names=("lazy", "working", "wealth"),
    )
    state_action_spaces = {0: _scs, 1: _scs}

    # ==================================================================================
    # create the Q_and_F functions
    # ==================================================================================

    def _Q_and_F(consumption, lazy, wealth, working, vf_arr, params):
        next_wealth = wealth + working - consumption
        next_lazy = lazy

        if vf_arr.size == 0:
            # this is the last period, when vf_arr = jnp.empty(0)
            expected_V = 0
        else:
            expected_V = map_coordinates(
                input=vf_arr[next_lazy],
                coordinates=jnp.array([next_wealth]),
            )

        U = consumption - 0.2 * lazy * working
        F = next_wealth >= 0

        Q = U + params["beta"] * expected_V

        return Q, F

    def _get_continuation_value(lazy, wealth, vf_arr):
        continuous_part = vf_arr[lazy]
        return map_coordinates(
            input=continuous_part,
            coordinates=jnp.array([wealth]),
        )

    max_Q_over_c = get_max_Q_over_c(
        Q_and_F=_Q_and_F,
        continuous_actions_names=("consumption",),
        states_and_discrete_actions_names=("lazy", "working", "wealth"),
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


def test_solve_brute_single_period_qc_values():
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
        states_and_discrete_actions_names=("a", "b", "c"),
    )

    def _Q_and_F(a, c, b, d, vf_arr, params):  # noqa: ARG001
        util = d
        feasib = d <= a + b + c
        return util, feasib

    max_Q_over_c = get_max_Q_over_c(
        Q_and_F=_Q_and_F,
        continuous_actions_names=("d",),
        states_and_discrete_actions_names=("a", "b", "c"),
    )

    expected = np.array([[[6.0, 7, 8], [7, 8, 9]], [[7, 8, 9], [8, 9, 10]]])

    # by setting max_Qc to identity, we can test that the max_Q_over_c function is
    # correctly applied to the state_action_space
    got = solve(
        params={},
        state_action_spaces={0: state_action_space},
        max_Q_over_c_functions={0: max_Q_over_c},
        max_Qc_functions={0: lambda x, params: x},  # noqa: ARG005
        logger=get_logger(debug_mode=False),
    )

    aaae(got[0], expected)
