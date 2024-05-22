import pytest
import timeit
import jax
import nvtx
from lcm.entry_point import get_lcm_function
from long_running import PARAMS, MODEL_CONFIG

SKIP_REASON = """The test is designed to run approximately 1 minute on a standard
laptop, such that we can differentiate the performance of running LCM on a GPU versus
on the CPU."""

params = PARAMS
def test_long():

    solve_model, template = get_lcm_function(MODEL_CONFIG, targets="solve")
    solve_model = jax.jit(solve_model)
    for i in range(10):
        with nvtx.annotate("solve_model", color="blue"):
            params['beta'] = params['beta'] - 0.01
            print(solve_model(params))

#jax.profiler.start_trace("/tmp/tensorboard")
test_long()
#jax.profiler.stop_trace()