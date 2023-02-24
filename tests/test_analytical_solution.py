"""
Implementation of analytical solution by Iskhakov et al (2017).
"""
import numpy as np
import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON_NO_BORROWING
from numpy.testing import assert_array_almost_equal as aaae
from scipy.optimize import root_scalar


def _u(c, d, delta):
    """
    Utility function.
    Args:
        c (float): consumption
        d (float): work indicator
        delta (float): disutility of work
    Returns:
        float: utility
    """
    if c > 0:
        return np.log(c) - d * delta
    else:
        return -np.inf


def _generate_policy_function_vector(wage, r, beta, tau):
    """
    Gererate consumption policy function vector given tau.

    Args:
        wage (float): income
        r (float): interest rate
        beta (float): discount factor
        tau (int): retirement age

    Returns:
        list: consumption policy function vector
    """

    policy_vec = [lambda m: m]

    # Generate liquidity constraint kink functions
    for i in range(1, tau + 1):
        policy_vec.append(
            lambda m, i=i: (
                m + wage * (np.sum([(1 + r) ** (-j) for j in range(1, i + 1)]))
            )
            / (np.sum([beta**j for j in range(0, i + 1)]))
        )

    # Generate retirement discontinuity functions
    for i in reversed(range(1, tau)):
        policy_vec.append(
            lambda m, i=i, tau=tau: (
                m + wage * (np.sum([(1 + r) ** (-j) for j in range(1, i + 1)]))
            )
            / (np.sum([beta**j for j in range(0, tau + 1)]))
        )

    policy_vec.append(
        lambda m, tau=tau: m / (np.sum([beta**j for j in range(0, tau + 1)]))
    )

    return policy_vec


def _compute_wealth_tresholds(v_prime, wage, r, beta, delta, tau, consumption_policy):
    """
    Return wealth treshold for consumption policy decision given tau.

    Args:
        V_prime (function): continuation value of value function
        wage (float): labor income
        r (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
    """
    # Liquidity constraint threshold
    wealth_thresholds = [-np.inf, wage / ((1 + r) * beta)]

    # Retirement threshold
    k = delta * np.sum([beta**j for j in range(0, tau + 1)]) ** (-1)
    ret_threshold = ((wage / (1 + r)) * np.exp(-k)) / (1 - np.exp(-k))

    # Other kinks and discontinuities
    for i in range(0, (tau - 1) * 2):
        c_l = consumption_policy[i + 1]
        c_u = consumption_policy[i + 2]
        root_fct = (
            lambda m, c_l=c_l, c_u=c_u: _u(c=c_l(m), d=1, delta=delta)
            - _u(c=c_u(m), d=1, delta=delta)
            + beta * v_prime((1 + r) * (m - c_l(m)) + wage)
            - beta * v_prime((1 + r) * (m - c_u(m)) + wage)
        )

        sol = root_scalar(
            root_fct,
            method="bisect",
            bracket=[wealth_thresholds[i + 1], ret_threshold],
            xtol=1e-15,
        )
        assert sol.converged
        wealth_thresholds.append(sol.root)

    # Add retirement threshold
    wealth_thresholds.append(ret_threshold)

    # Add upper bound s.t. wealth_threshold has same size as policy_vec
    wealth_thresholds.append(np.inf)

    return wealth_thresholds


def _evaluate_piecewise_conditions(m, wealth_thresholds):
    """
    Determine sub-function of policy function given M.

    Args:
        wealth_thresholds (list): list of wealth thresholds
    Returns:
        list: list of booleans
    """
    cond_list = [
        m >= lb and m < ub
        for lb, ub in zip(wealth_thresholds[:-1], wealth_thresholds[1:])
    ]
    return cond_list


def _construct_model(wage, r, num_periods, beta, delta):
    """
    Construct model given parameters via backward inducton.

    Args:
        wage (float): income
        r (float): interest rate
        T (int): Length of life
        beta (float): discount factor
        delta (float): disutility of work
    Returns:
        list: list of value functions
        list: list of consumption policy functions
    """

    c_pol = [None] * num_periods
    v = [None] * num_periods
    work_dec = [None] * num_periods

    for t in reversed(range(0, num_periods)):
        if t == num_periods - 1:
            v[t] = lambda m: np.log(m)
            c_pol[t] = lambda m: m
            work_dec[t] = lambda: 0
        else:
            # Generate consumption function
            policy_vec = _generate_policy_function_vector(
                wage, r, beta, tau=num_periods - t - 1
            )
            wt = _compute_wealth_tresholds(
                v[t + 1], wage, r, beta, delta, num_periods - t - 1, policy_vec
            )
            cond_list = lambda m, wt=wt: _evaluate_piecewise_conditions(  # noqa: E731
                m, wt
            )
            c_pol[
                t
            ] = lambda m, policy_vec=policy_vec, cond_list=cond_list: np.piecewise(
                x=m, condlist=cond_list(m), funclist=policy_vec
            )

            # Determine retirement decision
            work_dec[t] = (
                lambda m, ret_threshold=wt[-2]: True if m < ret_threshold else False
            )

            # Calculate V
            v[t] = lambda m, work=work_dec[t], c=c_pol[t], v_prime=v[t + 1]: _u(
                c=c(m), d=work(m), delta=delta
            ) + beta * v_prime((1 + r) * (m - c(m)) + wage * work(m))

    return v, c_pol, work_dec


def simulate_lc(
    num_periods,
    a,
    params,
):
    """
    Simulate life cycle given parameters.

    Args:
        wage (float): income
        r (float): interest rate
        T (int): Length of life
        beta (float): discount factor
        delta (float): disutility of work
        A (float): initial wealth
    Returns:
        list: list of value functions
        list: list of consumption policy functions
        list: list of wealth
    """

    # Unpack parameters
    beta = params["beta"]
    wage = params["next_wealth"]["wage"]
    r = params["next_wealth"]["interest_rate"]
    delta = params["utility"]["delta"]

    v, c_pol, work_dec = _construct_model(
        wage=wage, r=r, num_periods=num_periods, beta=beta, delta=delta
    )

    # Simulate life cycle
    consumption = []
    value = []
    m = np.zeros(num_periods)
    m[0] = a

    for v_func, c, d, t in zip(v, c_pol, work_dec, range(0, num_periods)):
        consumption.append(c(m[t]).item())
        value.append(v_func(m[t]))
        if t < num_periods - 1:
            m[t + 1] = (1 + r) * (m[t] - c(m[t])) + wage * d(m[t])

    return v, [consumption, value, m]


@pytest.fixture()
def params_analytical_solution():
    params = {
        "beta": 0.98,
        "utility": {
            "delta": 1.0,
        },
        "next_wealth": {"wage": 20.0, "interest_rate": 0.0},
        "next_wealth_constraint": np.NaN,
        "working": np.NaN,
    }
    v, life_cycle_choices = simulate_lc(**params)
    return v


def test_analytical_solution(params_analytical_solution):

    # Number of periods
    num_periods = 4

    # Specify grid
    wealth_grid_size = 1000
    wealth_grid_min = 1
    wealth_grid_max = 100
    grid_vals = np.linspace(1, 100, wealth_grid_size)

    # Analytical Solution
    v_function, life_cycle_choices = simulate_lc(
        num_periods=num_periods, a=30, params=params_analytical_solution
    )
    v_analytical = [
        np.array([v_function[t](vals) for vals in grid_vals])
        for t in range(0, num_periods)
    ]

    # Model
    model = PHELPS_DEATON_NO_BORROWING
    model["n_periods"] = num_periods
    model["choices"]["consumption"]["n_points"] = 10000
    model["states"]["wealth"]["start"] = wealth_grid_min
    model["states"]["wealth"]["stop"] = wealth_grid_max
    model["states"]["wealth"]["n_points"] = wealth_grid_size

    solve_model, params_template = get_lcm_function(model=model)
    v_numerical = solve_model(params=params_analytical_solution)

    aaae(v_analytical, v_numerical, decimal=3)
