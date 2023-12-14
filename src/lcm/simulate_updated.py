def forward_simulation(
    params,
    vf_arr_list,
    argsolve_continuous_problem,
    argsolve_discrete_problem,
    state_indexers,
    continuous_choice_grids,
    model,
    next_state,
    initial_states,
    logger,
    additional_targets=None,
    seed=12345,
):
    """Simulate the model forward in time.

    Goal:

    for t in periods:
        sim_scs, sim_choice_segments = create_sim_state_choice_space(sim_states)
        cont_policies, cont_solution = argsolve_cont_problem(scs, vf_arr, params)
        discrete_policies = argsolve_discrete_problem(cont_solution, sim_choice_segments)
        sim_states, sim_key = next_states(sim_states, cont_policies, discrete_policies, sim_key)

    """
    # Update the vf_arr_list
    # ----------------------------------------------------------------------------------
    # We drop the value function array for the first period, because it is not needed
    # for the simulation. This is because in the first period the agents only consider
    # the current utility and the value function of next period. Similarly, the last
    # value function array is not required, as the agents only consider the current
    # utility in the last period.
    # ==================================================================================
    vf_arr_list = vf_arr_list[1:] + [None]

    # Preparations
    # ==================================================================================
    n_periods = len(vf_arr_list)
    n_initial_states = len(next(iter(initial_states.values())))

    sparse_choice_variables = model.variable_info.query("is_choice & is_sparse").index

    # The following variables are updated during the forward simulation
    states = initial_states
    key = jax.random.PRNGKey(seed=seed)

    # Forward simulation
    # ==================================================================================
    _simulation_results = []

    for period in range(n_periods):
        # Create data state choice space
        # ------------------------------------------------------------------------------
        # Initial states are treated as sparse variables, so that the sparse variables
        # in the data-state-choice-space correspond to the feasible product of sparse
        # choice variables and initial states. The space has to be created in each
        # iteration because the states change over time.
        # ==============================================================================
        data_scs, data_choice_segments = create_data_scs(
            states=states,
            model=model,
        )

        # Compute objects dependent on data-state-choice-space
        # ==============================================================================
        dense_vars_grid_shape = tuple(
            len(grid) for grid in data_scs.dense_vars.values()
        )
        cont_choice_grid_shape = tuple(
            len(grid) for grid in continuous_choice_grids[period].values()
        )

        # Compute optimal continuous choice conditional on discrete choices
        # ==============================================================================
        ccv_policy, ccv = solve_continuous_problem(
            data_scs=data_scs,
            compute_ccv=argsolve_continuous_problem[period],
            continuous_choice_grids=continuous_choice_grids[period],
            vf_arr=vf_arr_list[period],
            state_indexers=state_indexers[period],
            params=params,
        )

        # Get optimal discrete choice given the optimal conditional continuous choices
        # ==============================================================================
        dense_argmax, sparse_argmax, value = argsolve_discrete_problem(
            conditional_continuation_value=ccv,
            choice_segments=data_choice_segments,
        )

        # Select optimal continuous choice corresponding to optimal discrete choice
        # ------------------------------------------------------------------------------
        # The conditional continuous choice argmax is computed for each discrete choice
        # in the data-state-choice-space. Here we select the the optimal continuous
        # choice corresponding to the optimal discrete choice (dense and sparse).
        # ==============================================================================
        cont_choice_argmax = filter_ccv_policy(
            ccv_policy,
            dense_argmax=dense_argmax,
            dense_vars_grid_shape=dense_vars_grid_shape,
        )
        if sparse_argmax is not None:
            cont_choice_argmax = cont_choice_argmax[sparse_argmax]

        # Convert optimal choice indices to actual choice values
        # ==============================================================================
        dense_choices = retrieve_non_sparse_choices(
            indices=dense_argmax,
            grids=data_scs.dense_vars,
            grid_shape=dense_vars_grid_shape,
        )

        cont_choices = retrieve_non_sparse_choices(
            indices=cont_choice_argmax,
            grids=continuous_choice_grids[period],
            grid_shape=cont_choice_grid_shape,
        )

        sparse_choices = {
            key: data_scs.sparse_vars[key][sparse_argmax]
            for key in sparse_choice_variables
        }

        # Store results
        # ==============================================================================
        choices = {**dense_choices, **sparse_choices, **cont_choices}

        _simulation_results.append(
            {
                "value": value,
                "choices": choices,
                "states": states,
            },
        )

        # Update states
        # ==============================================================================
        key, sim_keys = _generate_simulation_keys(
            key=key,
            ids=model.function_info.query("is_stochastic_next").index,
        )

        states = next_state(
            **states,
            **choices,
            _period=jnp.repeat(period, n_initial_states),
            params=params,
            keys=sim_keys,
        )

        # 'next_' prefix is added by the next_state function, but needs to be removed
        # because in the next period, next states are current states.
        states = {k.removeprefix("next_"): v for k, v in states.items()}

        logger.info("Period: %s", period)

    processed = _process_simulated_data(_simulation_results)

    if additional_targets is not None:
        calculated_targets = _compute_targets(
            processed,
            targets=additional_targets,
            model_functions=model.functions,
            params=params,
        )
        processed = {**processed, **calculated_targets}

    return _as_data_frame(processed, n_periods=n_periods)