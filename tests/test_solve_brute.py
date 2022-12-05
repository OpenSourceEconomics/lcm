import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from lcm.solve_brute import solve
from lcm.solve_brute import solve_continuous_problem
from lcm.state_space import Space
from numpy.testing import assert_array_almost_equal as aaae


def test_solve_brute():
    """Test solve brute with hand written inputs.

    Normally, these inputs would be created from a model specification. For now this
    can be seen as reference of what the functions that process a model specification
    need to produce.

    """
    # ==================================================================================
    # create the params
    # ==================================================================================
    params = {"beta": 0.9}

    # ==================================================================================
    # create the list of state_choice_spaces
    # ==================================================================================
    _scs = Space(
        sparse_vars={},
        dense_vars={
            # pick [0, 1] such that no label translation is needed
            # lazy is like a type, it influences utility but is not affected by choices
            "lazy": jnp.array([0, 1]),
            "working": jnp.array([0, 1]),
            # pick [0, 1, 2] such that no coordinate mapping is needed
            "wealth": jnp.array([0.0, 1.0, 2.0]),
        },
    )
    state_choice_spaces = [_scs] * 2

    # ==================================================================================
    # create list of value function evaluators
    # ==================================================================================
    def _vfe(lazy, wealth, working, vf_arr):
        continuous_part = vf_arr[lazy, working]
        value = map_coordinates(
            input=continuous_part,
            coordinates=wealth,
            order=1,
            mode="nearest",
        )
        return value

    value_function_evaluators = [_vfe] * 2

    # ==================================================================================
    # create the state_indexers (trivial because we do not have sparsity)
    # ==================================================================================
    state_indexers = [{}, {}]

    # ==================================================================================
    # create continuous choice grids
    # ==================================================================================

    # you 1 if working and have at most 2 in existing wealth, so
    _ccg = {"consumption": jnp.array([0, 1, 2, 3])}

    continuous_choice_grids = [_ccg] * 2

    # ==================================================================================
    # create the agent_functions that calculate utility, feasibility and next
    # ==================================================================================

    def _generic_agent_func(
        consumption, lazy, wealth, working, vf_arr, params  # noqa: U100
    ):
        """Calculate utility, feasibility and state transition.

        Args:
            consumption (float): Value of the continuous choice variable consumption.
            lazy (int): Value of the discrete state variable lazy
            wealth (float): Value of the continuous state variable wealth
            working (int): Value of the discrete choice variable working.

        Returns:
            float: The achieved utility
            bool: Whether the state choice combination is feasible
            dict: The next values of the state variables.

        """
        _u = consumption - 0.2 * lazy * working
        _next_wealth = wealth + working - consumption
        _next_lazy = lazy
        _feasible = _next_wealth >= 0
        return _u, _feasible, {"wealth": _next_wealth, "lazy": _next_lazy}

    def _last_period_agent_func(consumption, lazy, wealth, working, vf_arr, params):
        _u, _f, _ = _generic_agent_func(
            consumption, lazy, wealth, working, vf_arr, params
        )
        return _u, _f

    agent_functions = [_generic_agent_func, _last_period_agent_func]

    # ==================================================================================
    # create emax aggregators and choice segments
    # ==================================================================================
    choice_segments = [None, None]

    def calculate_emax(values, choice_segments, params):  # noqa: U100
        """Take max over axis that corresponds to working"""
        return values.max(axis=1)

    emax_calculators = [calculate_emax] * 2

    # ==================================================================================
    # call solve function
    # ==================================================================================

    solution = solve(
        params=params,
        state_choice_spaces=state_choice_spaces,
        value_function_evaluators=value_function_evaluators,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        agent_functions=agent_functions,
        emax_calculators=emax_calculators,
        choice_segments=choice_segments,
    )

    assert isinstance(solution, list)


def test_solve_continious_problem_no_vf_arr():
    state_choice_space = Space(
        dense_vars={"a": jnp.array([0, 1.0]), "b": jnp.array([2, 3.0])},
        sparse_vars={"c": jnp.array([4, 5, 6])},
    )

    def _utility_and_feasibility(a, c, b, d, vf_arr, params):  # noqa: U100
        util = d
        feasib = d <= a + b + c
        return util, feasib

    continuous_choice_grids = {"d": jnp.arange(12)}

    expected = np.array([[[6, 7, 8], [7, 8, 9]], [[7, 8, 9], [8, 9, 10]]])
    expected = np.transpose(expected, axes=(2, 0, 1))

    calculated = np.array(
        solve_continuous_problem(
            state_choice_space,
            _utility_and_feasibility,
            continuous_choice_grids,
            vf_arr=None,
            params={},
        )
    )

    aaae(calculated, expected)
