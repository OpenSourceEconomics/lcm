from functools import partial

from lcm.model_block import ModelBlock
from lcm.simulate import simulate
from lcm.solve_brute import solve


def get_lcm_function(
    model_specification,
    target,
):
    # Setup
    # ==================================================================================
    model = ModelBlock(model_specification)  # Larger models may consist of multiple
    # model blocks. Here we only require one.

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
        model.get_argsolve_continuous_problem(t, on="state_choice_space")
        for t in model.periods
    ]

    draw_next_states = [
        model.get_draw_next_state(t, on="state_choice") for t in model.periods
    ]

    # Partialling
    # ==================================================================================
    _solve_model = partial(
        solve,
        state_choice_spaces=state_choice_spaces,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_functions=solve_continuous_problem,
        emax_calculators=solve_discrete_problem,
    )

    _simulate_model = partial(
        simulate,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_policy_functions=argsolve_continuous_problem,
        model=model._specification,
        next_state=draw_next_states,
    )

    # Return the requested function
    # ==================================================================================
    targets = {
        "solve": _solve_model,
        "simulate": _simulate_model,
        "solve_and_simulate": partial(_simulate_model, solve_model=_solve_model),
    }
    return targets[target]
