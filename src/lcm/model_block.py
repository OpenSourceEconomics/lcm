from typing import Literal

from lcm.process_model import process_model


class ModelBlock:
    def __init__(self, specification: dict):
        self._specification = process_model(specification)
        self.periods = list(range(self._specification.n_periods))

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

    def get_solve_discrete_problem(self, period: int):
        """Reference: get_discrete_policy_calculator"""
        choice_segment = _get_choice_segments(...)

    def get_continuous_choice_grids(self, period: int):
        pass

    def get_draw_next_state(self, period: int, on: Literal["state_choice"]):
        """Reference: _get_next_state_function_simulation"""

    def get_create_data_state_choice_space(self, period: int):
        pass


def _get_choice_segments(*args, **kwargs):
    pass
