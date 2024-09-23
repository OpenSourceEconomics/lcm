import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates as jax_map_coordinates
from lcm.ndimage import map_coordinates as lcm_map_coordinates
import timeit
from functools import partial

# Sample input array and coordinates
input_array = np.random.uniform(-25, 100, 10_000)
coordinates = np.arange(-50, 150, 1)[None]

# Jitted functions
jax_jitted_map_coordinates = jax.jit(partial(jax_map_coordinates, order=1, cval=0))
lcm_jitted_map_coordinates = jax.jit(lcm_map_coordinates)

input_array_jax = jnp.asarray(input_array)
coordinates_jax = [jnp.asarray(c) for c in coordinates]

# Define wrapper functions for timeit
def time_jax():
    jax_jitted_map_coordinates(input_array_jax, coordinates_jax)

def time_lcm():
    lcm_jitted_map_coordinates(input_array_jax, coordinates_jax)

# Measure execution time
jax_time = timeit.timeit(time_jax, number=10_000)
lcm_time = timeit.timeit(time_lcm, number=10_000)

print(f"JAX map_coordinates time: {jax_time:.6f} seconds")
print(f"LCM map_coordinates time: {lcm_time:.6f} seconds")
