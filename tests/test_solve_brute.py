import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from lcm.interfaces import Space
from lcm.solve_brute import solve, solve_continuous_problem
from numpy.testing import assert_array_almost_equal as aaae


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
        value = map_coordinates(
            input=continuous_part,
            coordinates=jnp.array([wealth]),
            order=1,
            mode="nearest",
        )
        return value

    utility_and_feasibility_functions = [_utility_and_feasibility] * 2

    # ==================================================================================
    # create emax aggregators and choice segments
    # ==================================================================================
    choice_segments = [None, None]

    def calculate_emax(values, choice_segments, params):  # noqa: ARG001
        """Take max over axis that corresponds to working."""
        return values.max(axis=1)

    emax_calculators = [calculate_emax] * 2

    # ==================================================================================
    # call solve function
    # ==================================================================================

    solution = solve(
        params=params,
        state_choice_spaces=state_choice_spaces,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        utility_and_feasibility_functions=utility_and_feasibility_functions,
        emax_calculators=emax_calculators,
        choice_segments=choice_segments,
    )

    assert isinstance(solution, list)


def test_solve_continious_problem_no_vf_arr():
    state_choice_space = Space(
        dense_vars={"a": jnp.array([0, 1.0]), "b": jnp.array([2, 3.0])},
        sparse_vars={"c": jnp.array([4, 5, 6])},
    )

    def _utility_and_feasibility(a, c, b, d, vf_arr, params):  # noqa: ARG001
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
            state_indexers={},
            params={},
        ),
    )

    aaae(calculated, expected)
