import jax.numpy as jnp
import jax
import inspect
import functools
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from jaxlib import xla_client

def run_model():
    ### Choices
    working = jnp.array([0,1])
    consumption = jnp.linspace(1, 100, 100)
    exercise = jnp.linspace(0, 1, 200)
    ### States
    health = jnp.linspace(0, 1, 100)
    wealth = jnp.linspace(1, 100, 100)
    disutility_of_work = 0.05
    interest_rate = 0.05
    beta = 0.95
    params = {'beta': beta,'disutility_of_work': disutility_of_work, 'interest_rate':interest_rate }

    retirement_age = 65
    periods = retirement_age - 18

    def utility(consumption, working, health, exercise, disutility_of_work):
        return jnp.log(consumption) - (disutility_of_work - health) * working - exercise


    # --------------------------------------------------------------------------------------
    # Auxiliary variables
    # --------------------------------------------------------------------------------------
    def labor_income(wage, working):
        return wage * working


    def wage(age):
        return 1 + 0.1 * age

    # --------------------------------------------------------------------------------------
    # State transitions
    # --------------------------------------------------------------------------------------
    def next_wealth(wealth, consumption, labor_income, interest_rate):
        return (1 + interest_rate) * (wealth + labor_income - consumption)


    def next_health(health, exercise, working):
        return health * (1 + exercise - working / 2)


    # --------------------------------------------------------------------------------------
    # Constraints
    # --------------------------------------------------------------------------------------
    def consumption_constraint(consumption, wealth, labor_income):
        return consumption <= wealth + labor_income

    def u_and_f_last(consumption, working, health, exercise,wealth, period, vf_arr,params, last_period):
        age = period + 18
        income = labor_income(wage(age),working)
        return utility(consumption, working, health, exercise, params['disutility_of_work']), consumption_constraint(consumption, wealth, income)
    def u_and_f(consumption, working, health, exercise,wealth, period, vf_arr,params, last_period):
        age = period + 18
        income = labor_income(wage(age),working)
        step_length_wealth = (100 - 1) / (100 - 1)
        next_wealth_pos = (next_wealth(wealth, consumption, income, params['interest_rate']) - 1) / step_length_wealth
        step_length_health = (1 - 0) / (100 - 1)
        next_health_pos = (next_health(health, exercise, working) - 0) / step_length_health
        next_state = jnp.array([next_health_pos,next_wealth_pos])
        
        ccv = map_coordinates(vf_arr, list(next_state), order = 1, mode= 'nearest')
        big_u = utility(consumption, working, health, exercise, params['disutility_of_work']) + params['beta'] * ccv
        return big_u, consumption_constraint(consumption, wealth, income)
        
    def _base_productmap(func, product_axes: list[str]):
        signature = inspect.signature(func)
        parameters = list(signature.parameters)

        positions = [parameters.index(ax) for ax in product_axes]

        vmap_specs = []
        # We iterate in reverse order such that the output dimensions are in the same order
        # as the input dimensions.
        for pos in reversed(positions):
            spec = [None] * len(parameters)  # type: list[int | None]
            spec[pos] = 0
            vmap_specs.append(spec)

        vmapped = func
        for spec in vmap_specs:
            vmapped = vmap(vmapped, in_axes=spec)

        return vmapped

    vf_arr = jnp.zeros((len(wealth),len(health)))
    last_period = True
    reversed_solution = []
    for period in reversed(range(periods)):
        if last_period:
            utility_and_feasibility = _base_productmap(
                func=u_and_f_last,
                product_axes=['consumption','exercise'],
            )
        else:
            utility_and_feasibility = _base_productmap(
                func=u_and_f,
                product_axes=['consumption','exercise'],
            )
        @functools.wraps(utility_and_feasibility)
        def compute_ccv(*args, **kwargs):
            u, f = utility_and_feasibility(*args, **kwargs)
            return u.max(where=f, initial=-jnp.inf)
        cont_mapped = _base_productmap(compute_ccv, product_axes=['health','wealth','working'])
        jit_cont_mapped = jax.jit(cont_mapped)
        ccvs = jit_cont_mapped(consumption, working, health, exercise,wealth, period, vf_arr,params, last_period)
        vf_arr = ccvs.max(axis = 2,initial=-jnp.inf)
        reversed_solution.append(vf_arr)
        last_period = False
    return reversed_solution
jitted_run = jax.jit(run_model)
jitted = jitted_run.lower().compile().as_text()

def todotgraph(x):
   return xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(x))

with open("example.dot", "w") as f:
    f.write(jitted_run.lower().compiler_ir('hlo').as_hlo_dot_graph())
with open("example_compiled.dot", "w") as f:
    f.write(todotgraph(jitted))
for i in range(1):
    print(list(reversed(jitted_run())))


