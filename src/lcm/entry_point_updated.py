from lcm.model_block import ModelBlock
from lcm.solve_brute import solve
from lcm.simulate import simulate
from functools import partial


def get_lcm_function(
    model_specification,
):
    # Larger models will consist of multiple model blocks, here we only have one
    model = ModelBlock(model_specification)
    
    state_choice_spaces = [model.get_state_choice_space(t) for t in model.periods]
    
    state_indexers = [model.get_state_indexer(t) for t in model.periods]
    
    solve_continuous_problem = [
        model.get_solve_continuous_problem(t, on="state_choice_space")
        for t in model.periods
    ]
    
    solve_discrete_problem = [
        model.get_solve_discrete_problem(t) for t in model.periods
    ]
    
    continuous_choice_grids = [
        model.get_continuous_choice_grids(t) for t in model.periods
    ]
    
    draw_next_states = [
        model.get_draw_next_state(t, on="state_choice") for t in model.periods
    ]
    
    solve_model = partial(
        solve,
        state_choice_spaces=state_choice_spaces,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_functions=solve_continuous_problem,
        emax_calculators=solve_discrete_problem,
    )

    simulate_model = partial(
        simulate,
        state_indexers=state_indexers,
        continuous_choice_grids=continuous_choice_grids,
        compute_ccv_policy_functions=compute_ccv_policy_functions,
        model=_mod,
        next_state=draw_next_states,
    )
    
    return solve_model
