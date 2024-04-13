import inspect
from collections.abc import Callable
from typing import TypeVar

from jax import vmap

from lcm.functools import allow_args, allow_kwargs

F = TypeVar("F", bound=Callable)


def spacemap(
    func: F,
    dense_vars: list[str],
    sparse_vars: list[str],
    *,
    dense_first: bool,
) -> F:
    """Apply vmap such that func is evaluated on a space of dense and sparse variables.

    This is achieved by applying a product map for all dense_vars and a vmap for the
    sparse_vars.

    In contrast to vmap, spacemap preserves the function signature and allows the
    function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        dense_vars (list): Names of the dense variables, i.e. those that are simply
            stored as array of possible values in the grid because the possible values
            of those variable does not depend on the value of other variables.
        sparse_vars (list): Names of the sparse variables, i.e. those that are stored
            as arrays of possible combinations of variables in the grid because the
            possible values of these variables does depend on the value of other
            variables.
        dense_first (bool): Whether the dense or sparse dimensions should come first
            in the output of the dispatched function.


    Returns:
        callable: A callable with the same arguments as func (but with an additional
            leading dimension) that returns a jax.numpy.ndarray or pytree of arrays.
            If ``func`` returns a scalar, the dispatched function returns a
            jax.numpy.ndarray with k + 1 dimensions, where k is the length of
            ``dense_vars`` and the additional dimension corresponds to the
            ``sparse_vars``. The order of the dimensions is determined by the order of
            ``dense_vars`` as well as the ``dense_first`` argument.
            If the output of ``func`` is a jax pytree, the usual jax behavior applies,
            i.e. the leading dimensions of all arrays in the pytree are as described
            above but there might be additional dimensions.

    """
    # Check inputs
    # ==================================================================================
    if not set(dense_vars).isdisjoint(sparse_vars):
        raise ValueError("dense_vars and sparse_vars overlap.")

    if len(set(dense_vars)) < len(dense_vars):
        raise ValueError("Same argument provided more than once in dense variables.")

    if len(set(sparse_vars)) < len(sparse_vars):
        raise ValueError("Same argument provided more than once in sparse variables.")

    func = allow_args(func)  # vmap cannot deal with keyword-only arguments

    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    positions = [parameters.index(cv) for cv in sparse_vars]
    in_axes = [0 if p in positions else None for p in range(len(parameters))]

    if not sparse_vars:
        vmapped = _product_map(func, dense_vars)
    elif dense_first:
        vmapped = vmap(func, in_axes=in_axes)
        vmapped = _product_map(vmapped, dense_vars)
    else:
        vmapped = _product_map(func, dense_vars)
        vmapped = vmap(vmapped, in_axes=in_axes)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = signature  # type: ignore[attr-defined]

    return allow_kwargs(vmapped)


def productmap(func: F, variables: list[str]) -> F:
    """Apply vmap such that func is evaluated on the cartesian product of product_axes.

    This is achieved by an iterative application of vmap.

    In contrast to vmap, productmap preserves the function signature and allows the
    function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        variables (list): List with names of arguments that over which the cartesian
            product should be formed.

    Returns:
        callable: A callable with the same arguments as func (but with an additional
            leading dimension) that returns a jax.numpy.ndarray or pytree of arrays.
            If ``func`` returns a scalar, the dispatched function returns a
            jax.numpy.ndarray with k dimensions, where k is the length of ``variables``.
            The order of the dimensions is determined by the order of ``variables``
            which can be different to the order of ``funcs`` arguments. If the output of
            ``func`` is a jax pytree, the usual jax behavior applies, i.e. the leading
            dimensions of all arrays in the pytree are as described above but there
            might be additional dimensions.

    """
    func = allow_args(func)  # vmap cannot deal with keyword-only arguments

    if len(variables) != len(set(variables)):
        raise ValueError("Same argument provided more than once.")

    signature = inspect.signature(func)
    vmapped = _product_map(func, variables)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = signature  # type: ignore[attr-defined]

    return allow_kwargs(vmapped)


def vmap_1d(func: F, variables: list[str]) -> F:
    """Apply vmap such that func is mapped over the specified variables.

    In contrast to vmap, vmap_1d preserves the function signature and allows the
    function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        variables (list): List with names of arguments that over which we map.

    Returns:
        callable: A callable with the same arguments as func (but with an additional
            leading dimension) that returns a jax.numpy.ndarray or pytree of arrays.
            If ``func`` returns a scalar, the dispatched function returns a
            jax.numpy.ndarray with 1 dimension and length k, where k is the length of
            one of the mapped inputs in ``variables``. The order of the dimensions is
            determined by the order of ``variables`` which can be different to the order
            of ``funcs`` arguments. If the output of ``func`` is a jax pytree, the usual
            jax behavior applies, i.e. the leading dimensions of all arrays in the
            pytree are as described above but there might be additional dimensions.

    """
    if len(variables) != len(set(variables)):
        raise ValueError("Same argument provided more than once.")

    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    positions = [parameters.index(var) for var in variables]

    in_axes = [0 if p in positions else None for p in range(len(parameters))]

    vmapped = vmap(func, in_axes=in_axes)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = signature  # type: ignore[attr-defined]

    return allow_kwargs(vmapped)


def _product_map(func: F, product_axes: list[str] | None) -> F:
    """Do actual product map without signature changes.

    Args:
        func (callable): The function to be dispatched.
        product_axes (list): List with names of arguments over which we apply vmap.

    Returns:
        callable: A callable with the same arguments as func (but with an additional

    """
    signature = inspect.signature(func)
    parameters = list(signature.parameters)
    if product_axes is None:
        product_axes = parameters

    positions = [parameters.index(ax) for ax in product_axes]

    vmap_specs = []
    for pos in reversed(positions):
        spec = [None] * len(parameters)  # type: list[int | None]
        spec[pos] = 0
        vmap_specs.append(spec)

    vmapped = func
    for spec in vmap_specs:
        vmapped = vmap(vmapped, in_axes=spec)

    return vmapped
