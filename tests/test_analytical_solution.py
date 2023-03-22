"""
Implementation of analytical solution by Iskhakov et al (2017).
"""
from functools import partial

import numpy as np
import pytest
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON_NO_BORROWING
from numpy.testing import assert_array_almost_equal as aaae
from scipy.optimize import root_scalar


def _u(c, work_dec, delta):
    """
    Utility function.
    Args:
        c (float): consumption
        work_dec (float): work indicator (True or False)
        delta (float): disutility of work
    Returns:
        float: utility
    """
    if c > 0:
        return np.log(c) - work_dec * delta
    else:
        return -np.inf


def _generate_policy_function_vector(wage, r, beta, tau):
    """
    Gererate consumption policy function vector given tau.

    This function returns the functions that are used in the
    piecewise consumption function.
    Args:
        wage (float): income
        r (float): interest rate
        beta (float): discount factor
        tau (int): periods left until end of life

    Returns:
        dict: consumption policy dict
    """

    policy_vec_worker = [lambda m: m]

    # Generate liquidity constraint kink functions
    for i in range(1, tau + 1):
        policy_vec_worker.append(
            lambda m, i=i: (
                m + wage * (np.sum([(1 + r) ** (-j) for j in range(1, i + 1)]))
            )
            / (np.sum([beta**j for j in range(0, i + 1)]))
        )

    # Generate retirement discontinuity functions
    for i in reversed(range(1, tau)):
        policy_vec_worker.append(
            lambda m, i=i, tau=tau: (
                m + wage * (np.sum([(1 + r) ** (-j) for j in range(1, i + 1)]))
            )
            / (np.sum([beta**j for j in range(0, tau + 1)]))
        )
    policy_vec_worker.append(
        lambda m, tau=tau: m / (np.sum([beta**j for j in range(0, tau + 1)]))
    )

    # Generate function for retirees
    policy_retiree = lambda m, tau=tau: m / (  # noqa: E731
        np.sum([beta**j for j in range(0, tau + 1)])
    )

    return {"worker": policy_vec_worker, "retired": policy_retiree}


def _compute_wealth_tresholds(v_prime, wage, r, beta, delta, tau, consumption_policy):
    """
    Compute wealth treshold for piecewise consumption function.

    Args:
        v_prime (function): continuation value of value function
        wage (float): labor income
        r (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
        consumption_policy (list): consumption policy vector

    Returns:
        list: list of wealth thresholds
    """
    # Liquidity constraint threshold
    wealth_thresholds = [-np.inf, wage / ((1 + r) * beta)]

    # Retirement threshold
    k = delta * np.sum([beta**j for j in range(0, tau + 1)]) ** (-1)
    ret_threshold = ((wage / (1 + r)) * np.exp(-k)) / (1 - np.exp(-k))

    # Other kinks and discontinuities: Root finding
    for i in range(0, (tau - 1) * 2):
        c_l = consumption_policy[i + 1]
        c_u = consumption_policy[i + 2]
        root_fct = (
            lambda m, c_l=c_l, c_u=c_u: _u(c=c_l(m), work_dec=True, delta=delta)
            - _u(c=c_u(m), work_dec=True, delta=delta)
            + beta * v_prime((1 + r) * (m - c_l(m)) + wage, work_status=True)
            - beta * v_prime((1 + r) * (m - c_u(m)) + wage, work_status=True)
        )

        sol = root_scalar(
            root_fct,
            method="brentq",
            bracket=[wealth_thresholds[i + 1], ret_threshold],
            xtol=1e-10,
            rtol=1e-10,
            maxiter=1000,
        )
        assert sol.converged
        wealth_thresholds.append(sol.root)

    # Add retirement threshold
    wealth_thresholds.append(ret_threshold)

    # Add upper bound
    wealth_thresholds.append(np.inf)

    return wealth_thresholds


def _evaluate_piecewise_conditions(m, wealth_thresholds):
    """
    Determine correct sub-function of policy function given wealth m.

    Args:
        m (float): current wealth level
        wealth_thresholds (list): list of wealth thresholds
    Returns:
        list: list of booleans
    """
    cond_list = [
        m >= lb and m < ub
        for lb, ub in zip(wealth_thresholds[:-1], wealth_thresholds[1:])
    ]
    return cond_list


def work_decision(m, work_status, wealth_thresholds):
    """
    Determine work decision given current wealth level.

    Args:
        m (float): current wealth level
        work_status (bool): work status from last period
        wealth_thresholds (list): list of wealth thresholds
    Returns:
        bool: work decision
    """
    if work_status is not False:
        return True if m < wealth_thresholds[-2] else False
    else:
        return False


def _consumption(m, work_status, policy_dict, wt):
    """
    Determine consumption given current wealth level.

    Args:
        m (float): current wealth level
        work_status (bool): work status from last period
        policy_dict (dict): dictionary of consumption policy functions
        wt (list): list of wealth thresholds
    Returns:
        float: consumption
    """
    if work_status is False:
        cons = policy_dict["retired"](m)
        return cons
    else:
        condlist = _evaluate_piecewise_conditions(m, wealth_thresholds=wt)
        cons = np.piecewise(x=m, condlist=condlist, funclist=policy_dict["worker"])
        return cons


def _value_function(
    m, work_status, work_dec_func, c_pol, v_prime, beta, delta, tau, r, wage
):
    """
    Determine value function given current wealth level and retirement status.

    Args:
        m (float): current wealth level
        work_status (bool): work decision from last period
        work_dec_func (function): work decision function
        c_pol (function): consumption policy function
        v_prime (function): continuation value of value function
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
        r (float): interest rate
        wage (float): labor income
    Returns:
        float: value function
    """
    if m == 0:
        return -np.inf
    elif work_status is False:
        a = np.log(m) * np.sum([beta**j for j in range(0, tau + 1)])
        b = -np.log(np.sum([beta**j for j in range(0, tau + 1)]))
        c = np.sum([beta**j for j in range(0, tau + 1)])
        d = beta * (np.log(beta) + np.log(1 + r))
        e = np.sum(
            [
                beta**j * np.sum([beta**i for i in range(0, tau - j)])
                for j in range(0, tau)
            ]
        )
        v = a + b * c + d * e
    else:
        work_dec = work_dec_func(m=m, work_status=work_status)
        cons = c_pol(m=m, work_status=work_status)

        inst_util = _u(c=cons, work_dec=work_dec, delta=delta)
        cont_val = v_prime((1 + r) * (m - cons) + wage * work_dec, work_status=work_dec)

        v = inst_util + beta * cont_val

    return v


def _construct_model(delta, num_periods, param_dict):
    """
    Construct model given parameters via backward inducton.

    Args:
        delta (float): disutility of work
        num_periods (int): length of life
        param_dict (dict): dictionary of parameters
    Returns:
        list: list of value functions
    """

    c_pol = [None] * num_periods
    v = [None] * num_periods
    work_dec_func = [None] * num_periods

    for t in reversed(range(0, num_periods)):
        if t == num_periods - 1:
            v[t] = lambda m, work_status: np.log(m) if m > 0 else -np.inf  # noqa: U100
            c_pol[t] = lambda m, work_status: m  # noqa: U100
            work_dec_func[t] = lambda m, work_status: False  # noqa: U100
        else:
            # Time left until retirement
            param_dict["tau"] = num_periods - t - 1

            # Generate consumption function
            policy_dict = _generate_policy_function_vector(**param_dict)

            wt = _compute_wealth_tresholds(
                v_prime=v[t + 1],
                consumption_policy=policy_dict["worker"],
                delta=delta,
                **param_dict,
            )

            c_pol[t] = partial(_consumption, policy_dict=policy_dict, wt=wt)

            # Determine retirement status
            work_dec_func[t] = partial(
                work_decision,
                wealth_thresholds=wt,
            )

            # Calculate V
            v[t] = partial(
                _value_function,
                work_dec_func=work_dec_func[t],
                c_pol=c_pol[t],
                v_prime=v[t + 1],
                delta=delta,
                **param_dict,
            )
    return v


def analytical_solution(grid, beta, wage, r, delta, num_periods):
    """
    Compute value function analytically on a grid.

    Args:
        grid (list): grid of wealth levels
        beta (float): discount factor
        wage (float): labor income
        r (float): interest rate
        delta (float): disutility of work
        num_periods (int): length of life
    Returns:
        list: values of value function
    """

    # Unpack parameters

    param_dict = {
        "beta": beta,
        "wage": wage,
        "r": r,
        "tau": None,
    }

    v_fct = _construct_model(
        delta=delta, num_periods=num_periods, param_dict=param_dict
    )

    v = np.array(
        [
            [
                list(map(v_fct[t], grid, [work_status] * len(grid)))
                for work_status in [True]
            ]
            for t in range(0, num_periods)
        ]
    )[:, 0, :]

    return v


@pytest.fixture()
def params_analytical_solution():
    params = {
        "beta": 0.98,
        "delta": 0.3,
        "wage": float(10),
        "r": 0.0,
        "num_periods": 3,
    }
    return params


def test_analytical_solution(params_analytical_solution):

    # Specify grid
    wealth_grid_size = 8_000
    wealth_grid_min = 1
    wealth_grid_max = 100
    grid_vals = np.linspace(wealth_grid_min, wealth_grid_max, wealth_grid_size)

    # Analytical Solution
    v_analytical = analytical_solution(grid=grid_vals, **params_analytical_solution)

    # Numerical Solution
    model = PHELPS_DEATON_NO_BORROWING
    model["n_periods"] = params_analytical_solution["num_periods"]
    model["choices"]["consumption"]["start"] = wealth_grid_min
    model["choices"]["consumption"]["stop"] = wealth_grid_max
    model["choices"]["consumption"]["n_points"] = wealth_grid_size
    model["states"]["wealth"]["start"] = wealth_grid_min
    model["states"]["wealth"]["stop"] = wealth_grid_max
    model["states"]["wealth"]["n_points"] = wealth_grid_size

    solve_model, params_template = get_lcm_function(model=model)

    params_template["beta"] = params_analytical_solution["beta"]
    params_template["labor_income"]["wage"] = params_analytical_solution["wage"]
    params_template["next_wealth"]["interest_rate"] = params_analytical_solution["r"]
    params_template["utility"]["delta"] = params_analytical_solution["delta"]
    v_numerical = np.array(solve_model(params=params_template))[:, 0, :]

    aaae(x=v_analytical, y=v_numerical, decimal=6)
