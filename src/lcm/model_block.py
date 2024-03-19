import functools
from typing import Literal

import jax.numpy as jnp

from lcm.argmax import argmax
from lcm.discrete_emax import get_emax_calculator
from lcm.dispatchers import productmap
from lcm.interfaces import Model
from lcm.model_functions import get_utility_and_feasibility_function
from lcm.process_model import process_model
from lcm.state_space import create_state_choice_space


class ModelBlock:

    def __init__(self, specification: dict):
        self._specification = process_model(specification)
        self.periods = list(range(self._specification.n_periods))
        self.last_period = self.periods[-1]

        # Store state-space related information
        # TODO(@timmens) This code needs to be restructured to fit better into the
        # period-based getter functions.
        state_choice_spaces, space_infos, state_indexers, choice_segments = (
            _get_state_choice_space_info(
                model=self._specification,
                periods=self.periods,
            )
        )
        self._state_choice_spaces = state_choice_spaces
        self._space_infos = space_infos
        self._state_indexers = state_indexers
        self._choice_segments = choice_segments

    def is_last_period(self, period: int):
        return period == self.last_period

    # ----------------------------------------------------------------------------------
    # State-space related getters
    # ----------------------------------------------------------------------------------

    def get_state_choice_space(self, period: int):
        return self._state_choice_spaces[period]

    def get_state_indexer(self, period: int):
        return self._state_indexers[period]

    def get_state_space_info(self, period: int):
        return self._space_infos[period]

    def get_choice_segments(self, period: int):
        return self._choice_segments[period]

    # Unused
    def get_state_space(self, period: int):
        pass

    def get_state_choice_indexer(self, period: int):
        pass

    def get_state_choice_space_info(self, period: int):
        pass

    # ----------------------------------------------------------------------------------
    # Utility functions
    # ----------------------------------------------------------------------------------

    def get_utility_and_feasibility(self, period: int):
        return get_utility_and_feasibility_function(
            model=self._specification,
            space_info=self.get_state_space_info(period),
            data_name="vf_arr",
            interpolation_options={},
            period=period,
            is_last_period=self.is_last_period(period),
        )

    def get_solve_continuous_problem(
        self,
        period: int,
        on: Literal["state_choice", "state_choice_space"],
    ):
        """Reference: create_compute_conditional_continuation_value"""
        return _create_compute_conditional_continuation_value(
            utility_and_feasibility=self.get_utility_and_feasibility(period),
            continuous_choice_variables=list(self.get_continuous_choice_grids(period)),
        )

    def get_argsolve_continuous_problem(
        self,
        period: int,
        on: Literal["state_choice", "state_choice_space"],
    ):
        """Reference: create_compute_conditional_continuation_policy"""
        return _create_compute_conditional_continuation_policy(
            utility_and_feasibility=self.get_utility_and_feasibility(period),
            continuous_choice_variables=list(self.get_continuous_choice_grids(period)),
        )

    def get_solve_discrete_problem(self, period: int):
        """Reference: get_emax_calculator"""
        return get_emax_calculator(
            shock_type=None,
            variable_info=self._specification.variable_info,
            is_last_period=self.is_last_period(period),
            choice_segments=self.get_choice_segments(period),
            params=self._specification.params,
        )

    def get_argsolve_discrete_problem(self, period: int):
        """Reference: get_discrete_policy_calculator, but depends on choice_segments."""

    def get_continuous_choice_grids(self, period: int):
        # Currently only time-invariant choice grids are supported, i.e., the period
        # argument has no effect.
        return _get_continuous_choice_grids(self._specification, period=period)

    def get_draw_next_state(self, period: int, on: Literal["state_choice"]):
        """Reference: _get_next_state_function_simulation"""

    def get_create_data_state_choice_space(self, period: int):
        pass

    # ----------------------------------------------------------------------------------
    # Unused
    # ----------------------------------------------------------------------------------

    def get_current_utility(
        self,
        period: int,
        on: Literal["state_choice", "state_choice_space"],
    ):
        pass

    def get_value_function(self, period: int, on: Literal["state", "state_space"]):
        pass

    def get_total_utility(
        self,
        period: int,
        on: Literal["state_choice", "state_choice_space"],
    ):
        pass

    def get_feasibility(
        self,
        period: int,
        on: Literal["state_choice", "state_choice_space"],
    ):
        pass


def _get_state_choice_space_info(
    model,
    periods,
):
    state_choice_spaces = []
    space_infos = []
    state_indexers = []
    choice_segments = []

    for period in periods:

        is_last_period = period == periods[-1]

        state_choice_space, space_info, state_indexer, choice_segment = (
            create_state_choice_space(
                model=model,
                period=period,
                is_last_period=is_last_period,
                jit_filter=False,
            )
        )
        state_choice_spaces.append(state_choice_space)
        space_infos.append(space_info)
        state_indexers.append(state_indexer)
        choice_segments.append(choice_segment)

    # Shift space info (in period t we require the space info of period t+1)
    space_infos = space_infos[1:] + [{}]

    return state_choice_spaces, space_infos, state_indexers, choice_segments


def _get_continuous_choice_grids(model: Model, period: int) -> dict[str, jnp.ndarray]:
    """Extract continuous choice grids from a model.

    Currently only time-invariant choice grids are supported, i.e., the period argument
    has no effect.

    Args:
        model (Model): Processed user model.
        period (int): Period for which to extract the continuous choice grids.

    Returns:
        dict[str, jnp.ndarray]: Dictionary with continuous choice grids, that maps names
            of continuous choice variables to grids of feasible values for that
            variable.

    """
    continuous_choice_vars = model.variable_info.query(
        "is_continuous & is_choice",
    ).index.tolist()
    return {var: model.grids[var] for var in continuous_choice_vars}


def _create_compute_conditional_continuation_value(
    utility_and_feasibility,
    continuous_choice_variables,
):
    """Create a function that computes the conditional continuation value.

    Note:
    -----
    This function solves the continuous choice problem conditional on a state-
    (discrete-)choice combination.

    Args:
        utility_and_feasibility (callable): A function that takes a state-choice
            combination and return the utility of that combination (float) and whether
            the state-choice combination is feasible (bool).
        continuous_choice_variables (list): List of choice variable names that are
            continuous.

    Returns:
        callable: A function that takes a state-choice combination and returns the
            conditional continuation value over the continuous choices.

    """
    if continuous_choice_variables:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_choice_variables,
        )

    @functools.wraps(utility_and_feasibility)
    def compute_ccv(*args, **kwargs):
        u, f = utility_and_feasibility(*args, **kwargs)
        return u.max(where=f, initial=-jnp.inf)

    return compute_ccv


def _create_compute_conditional_continuation_policy(
    utility_and_feasibility,
    continuous_choice_variables,
):
    """Create a function that computes the conditional continuation policy.

    Note:
    -----
    This function solves the continuous choice problem conditional on a state-
    (discrete-)choice combination.

    Args:
        utility_and_feasibility (callable): A function that takes a state-choice
            combination and return the utility of that combination (float) and whether
            the state-choice combination is feasible (bool).
        continuous_choice_variables (list): List of choice variable names that are
            continuous.

    Returns:
        callable: A function that takes a state-choice combination and returns the
            conditional continuation value over the continuous choices, and the index
            that maximizes the conditional continuation value.

    """
    if continuous_choice_variables:
        utility_and_feasibility = productmap(
            func=utility_and_feasibility,
            variables=continuous_choice_variables,
        )

    @functools.wraps(utility_and_feasibility)
    def compute_ccv_policy(*args, **kwargs):
        u, f = utility_and_feasibility(*args, **kwargs)
        _argmax, _max = argmax(u, where=f, initial=-jnp.inf)
        return _argmax, _max

    return compute_ccv_policy
