from functools import partial
from typing import Literal, get_args

from lcm.model_block import ModelBlock
from lcm.simulate_updated import forward_simulation
from lcm.solve_updated import backward_induction

ValidTargets = Literal["solve", "simulate", "solve_and_simulate"]


def get_lcm_function(
    model_specification: dict,
    target: ValidTargets,
):
    if target not in get_args(ValidTargets):
        raise ValueError(f"Invalid target: {target}. Must be in {ValidTargets}.")

    # Setup
    # ==================================================================================
    # Larger models may consist of multiple model blocks. Here we only require one.
    model = ModelBlock(model_specification)

    # Objects related to the state choice space
    # ==================================================================================
    state_choice_spaces = [model.get_state_choice_space(t) for t in model.periods]

    state_indexers = [model.get_state_indexer(t) for t in model.periods]

    continuous_choice_grids = [
        model.get_continuous_choice_grids(t) for t in model.periods
    ]

    # Functions that solve the agent's problem
    # ==================================================================================
    solve_continuous_problem = [
        model.get_solve_continuous_problem(t, on="state_choice_space")
        for t in model.periods
    ]

    solve_discrete_problem = [
        model.get_solve_discrete_problem(t) for t in model.periods
    ]

    # Functions that simulate the agent's choices
    # ==================================================================================
    argsolve_continuous_problem = [
        model.get_argsolve_continuous_problem(t, on="sim_state_choice_space")
        for t in model.periods
    ]

    argsolve_discrete_problem = [
        model.get_argsolve_discrete_problem(t) for t in model.periods
    ]

    draw_next_states = [
        model.get_draw_next_state(t, on="state_choice") for t in model.periods
    ]

    # Partialling
    # ==================================================================================
    _solve_model = partial(
        backward_induction,
        solve_continuous_problem=solve_continuous_problem,
        solve_discrete_problem=solve_discrete_problem,
        continuous_choice_grids=continuous_choice_grids,
        state_choice_spaces=state_choice_spaces,
        state_indexers=state_indexers,
    )

    _simulate_model = partial(
        forward_simulation,
        argsolve_continuous_problem=argsolve_continuous_problem,
        argsolve_discrete_problem=argsolve_discrete_problem,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        model=model._specification,
        next_state=draw_next_states,
    )

    # Return the requested function
    # ==================================================================================
    target_functions = {
        "solve": _solve_model,
        "simulate": _simulate_model,
        "solve_and_simulate": partial(_simulate_model, solve_model=_solve_model),
    }
    return target_functions[target]
