import jax.numpy as jnp
from lcm.example_models import PHELPS_DEATON, phelps_deaton_utility
from lcm.model_functions import get_utility_and_feasibility_function
from lcm.process_model import process_model
from lcm.state_space import create_state_choice_space
from numpy.testing import assert_array_equal


def test_get_utility_and_feasibility_function():
    model = process_model(PHELPS_DEATON)

    params = {
        "beta": 1.0,
        "utility": {"delta": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    _, space_info, _, _ = create_state_choice_space(
        model=model,
        period=0,
        jit_filter=False,
    )

    u_and_f = get_utility_and_feasibility_function(
        model=model,
        space_info=space_info,
        data_name="vf_arr",
        interpolation_options={},
        is_last_period=True,
    )

    consumption = jnp.array([10, 20, 30])
    retirement = jnp.array([0, 1, 0])
    wealth = jnp.array([20, 20, 20])

    u, f = u_and_f(
        consumption=consumption,
        retirement=retirement,
        wealth=wealth,
        params=params,
        vf_arr=None,
    )

    assert_array_equal(
        u,
        phelps_deaton_utility(
            consumption=consumption,
            working=1 - retirement,
            delta=1.0,
        ),
    )
    assert_array_equal(f, jnp.array([True, True, False]))
