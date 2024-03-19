from typing import Literal

import jax.numpy as jnp

from lcm.interfaces import Model
from lcm.process_model import process_model


class ModelBlock:
    def __init__(self, specification: dict):
        self._specification = process_model(specification)

        self.periods = list(range(self._specification.n_periods))
        self.last_period = self.periods[-1]

    def get_state_space(self, period: int):
        pass

    def get_state_choice_space(self, period: int):
        pass

    def get_state_indexer(self, period: int):
        pass

    def get_state_choice_indexer(self, period: int):
        pass

    def get_state_space_info(self, period: int):
        pass

    def get_state_choice_space_info(self, period: int):
        pass

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

    def get_solve_continuous_problem(
        self,
        period: int,
        on: Literal["state_choice", "state_choice_space"],
    ):
        """Reference: create_compute_conditional_continuation_value"""

    def get_argsolve_continuous_problem(
        self,
        period: int,
        on: Literal["state_choice", "state_choice_space"],
    ):
        """Reference: create_compute_conditional_continuation_policy"""

    def get_solve_discrete_problem(self, period: int):
        """Reference: get_emax_calculator"""
        choice_segment = _get_choice_segments(...)

    def get_argsolve_discrete_problem(self, period: int):
        """Reference: get_discrete_policy_calculator, but depends on choice_segments."""
        choice_segment = _get_choice_segments(...)

    def get_continuous_choice_grids(self, period: int):
        # Currently only time-invariant choice grids are supported, i.e., the period
        # argument has no effect.
        return _get_continuous_choice_grids(self._specification, period=period)

    def get_draw_next_state(self, period: int, on: Literal["state_choice"]):
        """Reference: _get_next_state_function_simulation"""

    def get_create_data_state_choice_space(self, period: int):
        pass


def _get_choice_segments(*args, **kwargs):
    pass


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
