from lcm.entry_point import get_lcm_function
from lcm.model_parttime_human_capital import (
    PARTTIME_HUMAN_CAPITAL,
    PARTTIME_HUMAN_CAPITAL_PARAMS,
)


def test_model_parttime_human_capital_solve():
    solve, params = get_lcm_function(model=PARTTIME_HUMAN_CAPITAL, targets="solve")

    solve(params=PARTTIME_HUMAN_CAPITAL_PARAMS)
