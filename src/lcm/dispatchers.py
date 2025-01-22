import inspect
from collections.abc import Callable
from typing import Literal, TypeVar
from jax import Array, vmap
from jax import numpy as jnp
from jax.lax import map
from lcm.functools import allow_args, allow_only_kwargs
from lcm.typing import ParamsDict

F = TypeVar("F", bound=Callable[..., Array])


def spacemap(
    func: F,
    state_and_discrete_vars: dict[str:Array],
    continous_vars: dict[str:Array],
    memory_restriction: int,
    params: ParamsDict,
    vf_arr: Array,
) -> F:
    """
    Evaluate func along all state and discrete choice axes in a way that reduces the memory usage 
    below a preset value, trying to find a balance between speed and memory usage.

    If the dimension of the state-choice space fits into memory this will be the same as vmapping along all axes.
    Otherwise the computation along the outermost state axes will be serialized until the remeining problem fits 
    into memory.

    This only works if the continous part of the model fits into memory, as this part has already been vmapped in func.
    To serialize parts of the continous axes one would have to write a more complicated function using scan that
    replicates the behaviour of map with the batch size parameter. For models with many continous axes there
    might be better ways to find the maximum along those axes than evaluating func on all points.


    Args:
        func: The function to be dispatched.
        state_and_discrete_vars: Dict of names and values for each discrete choice axis and state axis.
        continous_vars: Dict of names and values for each continous choice axis.
        memory_restriction: Maximum allowed memory usage of the vmap in Bytes. Maybe the user should be able to set this.
        Could also be grabbed through jax, but then we would have to set the limit very low 
        and users would not be able to overwrite it for better performance.
        params: Parameters of the Model.
        vf_arr: Discretized Value Function from previous period.


    Returns:
        A callable that evaluates func along the provided dicrete choice and state axes.

    """
    # Check inputs and prepare function
    # ==================================================================================
    
    # jax.vmap cannot deal with keyword-only arguments
    func = allow_args(func)
    
    # I removed duplicate and overlap checks because we are passing dicts now 
    # and overlap between state+dicrte and continous seems unlikely

    # Set the batch size parameters for the stacked maps, controlling the degree of serialization.
    # Checks if vmapping along given axis is possible, starting from the innermost discrete choice axis.
    # If vmapping is possible batch size is set to len(axis) because vamp=map if batch size=len(axis), 
    # otherwise batchsize is set to 1 serializing the evaluation of func along this axis.

    memory_strat = {}
    memory_restriction = (memory_restriction/4)/2
    for key in continous_vars.keys():
        memory_restriction = memory_restriction / jnp.size(continous_vars[key])
    for key in state_and_discrete_vars.keys():
        memory_restriction = memory_restriction/jnp.size(state_and_discrete_vars[key])
        if memory_restriction > 1:
            memory_strat[key] = jnp.size(state_and_discrete_vars[key])
        else:
            memory_strat[key] = 1
        
    mapped = _base_productmap_map(func, state_and_discrete_vars, continous_vars, memory_strat,params, vf_arr)
    
    return mapped


def vmap_1d(
    func: F,
    variables: list[str],
    *,
    callable_with: Literal["only_args", "only_kwargs"] = "only_kwargs",
) -> F:
    """Apply vmap such that func is mapped over the specified variables.

    In contrast to a general vmap call, vmap_1d vectorizes along the leading axis of all
    of the requested variables simultaneously. Moreover, it preserves the function
    signature and allows the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched.
        variables: List with names of arguments that over which we map.
        callable_with: Whether to apply the allow_kwargs decorator to the dispatched
            function. If "only_args", the returned function can only be called with
            positional arguments. If "only_kwargs", the returned function can only be
            called with keyword arguments.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns a jax.numpy.ndarray or pytree of arrays. If ``func``
        returns a scalar, the dispatched function returns a jax.numpy.ndarray with 1
        jax.numpy.ndarray with 1 dimension and length k, where k is the length of one of
        the mapped inputs in ``variables``. The order of the dimensions is determined by
        the order of ``variables`` which can be different to the order of ``funcs``
        arguments. If the output of ``func`` is a jax pytree, the usual jax behavior
        applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    duplicates = {v for v in variables if variables.count(v) > 1}
    if duplicates:
        raise ValueError(
            f"Same argument provided more than once in variables: {duplicates}",
        )

    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    positions = [parameters.index(var) for var in variables]

    # Create in_axes to apply vmap over variables. This has one entry for each argument
    # of func, indicating whether the argument should be mapped over or not. None means
    # that the argument should not be mapped over, 0 means that it should be mapped over
    # the leading axis of the input.
    in_axes_for_vmap = [None] * len(parameters)  # type: list[int | None]
    for p in positions:
        in_axes_for_vmap[p] = 0

    vmapped = vmap(func, in_axes=in_axes_for_vmap)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = signature  # type: ignore[attr-defined]

    if callable_with == "only_kwargs":
        out = allow_only_kwargs(vmapped)
    elif callable_with == "only_args":
        out = vmapped
    else:
        raise ValueError(
            f"Invalid callable_with option: {callable_with}. Possible options are "
            "('only_args', 'only_kwargs')",
        )

    return out


def productmap(func: F, variables: list[str]) -> F:
    """Apply vmap such that func is evaluated on the Cartesian product of variables.

    This is achieved by an iterative application of vmap.

    In contrast to _base_productmap, productmap preserves the function signature and
    allows the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched.
        variables: List with names of arguments that over which the Cartesian product
            should be formed.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns a jax.numpy.ndarray or pytree of arrays. If ``func``
        returns a scalar, the dispatched function returns a jax.numpy.ndarray with k
        dimensions, where k is the length of ``variables``. The order of the dimensions
        is determined by the order of ``variables`` which can be different to the order
        of ``funcs`` arguments. If the output of ``func`` is a jax pytree, the usual jax
        behavior applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    func = allow_args(func)  # jax.vmap cannot deal with keyword-only arguments

    duplicates = {v for v in variables if variables.count(v) > 1}
    if duplicates:
        raise ValueError(
            f"Same argument provided more than once in variables: {duplicates}",
        )

    signature = inspect.signature(func)
    vmapped = _base_productmap(func, variables)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = signature  # type: ignore[attr-defined]

    return allow_only_kwargs(vmapped)


def _base_productmap_map(func: F, state_and_discrete_vars: dict[str:Array], continous_vars: dict[str:Array], strat, params, vf_arr) -> F:
    """Map func over the Cartesian product of state_and_discrete_vars.

    All arguments needed for the evaluation of func are passed via keyword args upon creation, 
    the returned callable takes no arguments.

    Args:
        func: The function to be dispatched.
        state_and_discrete_vars: Dict of names and values for each discrete choice axis and state axis.
        continous_vars: Dict of names and values for each continous choice axis.
        params: Parameters of the Model.
        vf_arr: Discretized Value Function from previous period.

    Returns:
        A callable that maps func over the provided axes.

    """
    mapped = lambda **vals : func(**continous_vars, vf_arr = vf_arr, params = params, **vals)
    def stack_maps(func, var, axis):
        def one_more(**xs):
            return map(lambda x_i: func(**xs, **{var:x_i}), axis, batch_size=strat[var])
        return one_more
    for key,value in reversed(state_and_discrete_vars.items()):
        mapped = stack_maps(mapped,key,value)
    

    return mapped

def _base_productmap(func: F, product_axes: list[Array]) -> F:
    """Map func over the Cartesian product of product_axes.

    Like vmap, this function does not preserve the function signature and does not allow
    the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched. Cannot have keyword-only arguments.
        product_axes: List with names of arguments over which we apply vmap.

    Returns:
        A callable with the same arguments as func. See ``product_map`` for details.

    """
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
