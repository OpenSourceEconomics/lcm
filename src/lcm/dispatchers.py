import inspect
from collections.abc import Callable
from typing import TypeVar

from jax import vmap

from lcm.functools import allow_args, allow_only_kwargs

F = TypeVar("F", bound=Callable)


def spacemap(
    func: F,
    dense_vars: list[str],
    sparse_vars: list[str],
    *,
    put_dense_first: bool,
) -> F:
    """Apply vmap such that func is evaluated on a space of dense and sparse variables.

    This is achieved by applying _base_productmap for all dense variables and vmap_1d
    for the sparse variables.

    spacemap preserves the function signature and allows the function to be called with
    keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        dense_vars (list): Names of the dense variables, i.e. those that are stored
            as arrays of possible values in the grid.
        sparse_vars (list): Names of the sparse variables, i.e. those that are stored
            as arrays of possible combinations of variables in the grid.
        put_dense_first (bool): Whether the dense or sparse dimensions should come first
            in the output of the dispatched function.


    Returns:
        callable: A callable with the same arguments as func (but with an additional
            leading dimension) that returns a jax.numpy.ndarray or pytree of arrays.
            If ``func`` returns a scalar, the dispatched function returns a
            jax.numpy.ndarray with k + 1 dimensions, where k is the length of
            ``dense_vars`` and the additional dimension corresponds to the
            ``sparse_vars``. The order of the dimensions is determined by the order of
            ``dense_vars`` as well as the ``put_dense_first`` argument.
            If the output of ``func`` is a jax pytree, the usual jax behavior applies,
            i.e. the leading dimensions of all arrays in the pytree are as described
            above but there might be additional dimensions.

    """
    # Check inputs and prepare function
    # ==================================================================================
    if overlap := set(dense_vars).intersection(sparse_vars):
        raise ValueError(
            f"Dense and sparse variables must be disjoint. Overlap: {overlap}",
        )

    if duplicates := {v for v in dense_vars if dense_vars.count(v) > 1}:
        raise ValueError(
            f"Same argument provided more than once in dense variables: {duplicates}",
        )

    if duplicates := {v for v in sparse_vars if sparse_vars.count(v) > 1}:
        raise ValueError(
            f"Same argument provided more than once in sparse variables: {duplicates}",
        )

    # jax.vmap cannot deal with keyword-only arguments
    func = allow_args(func)

    # Apply vmap_1d for sparse and _base_productmap for dense variables
    # ==================================================================================
    if not sparse_vars:
        vmapped = _base_productmap(func, dense_vars)
    elif put_dense_first:
        vmapped = vmap_1d(func, variables=sparse_vars, apply_allow_kwargs=False)
        vmapped = _base_productmap(vmapped, dense_vars)
    else:
        vmapped = _base_productmap(func, dense_vars)
        vmapped = vmap_1d(vmapped, variables=sparse_vars, apply_allow_kwargs=False)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]

    return allow_only_kwargs(vmapped)


def vmap_1d(func: F, variables: list[str], *, apply_allow_kwargs: bool = True) -> F:
    """Apply vmap such that func is mapped over the specified variables.

    In contrast to a general vmap call, vmap_1d vectorizes along the leading axis of all
    of the requested variables simultaneously. Moreover, it preserves the function
    signature and allows the function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        variables (list): List with names of arguments that over which we map.
        apply_allow_kwargs (bool): Whether to apply the allow_kwargs decorator to the
            dispatched function.

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
    if duplicates := {v for v in variables if variables.count(v) > 1}:
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

    return allow_only_kwargs(vmapped) if apply_allow_kwargs else vmapped


def productmap(func: F, variables: list[str]) -> F:
    """Apply vmap such that func is evaluated on the Cartesian product of variables.

    This is achieved by an iterative application of vmap.

    In contrast to _base_productmap, productmap preserves the function signature and
    allows the function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        variables (list): List with names of arguments that over which the Cartesian
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
    func = allow_args(func)  # jax.vmap cannot deal with keyword-only arguments

    if duplicates := {v for v in variables if variables.count(v) > 1}:
        raise ValueError(
            f"Same argument provided more than once in variables: {duplicates}",
        )

    signature = inspect.signature(func)
    vmapped = _base_productmap(func, variables)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = signature  # type: ignore[attr-defined]

    return allow_only_kwargs(vmapped)


def _base_productmap(func: F, product_axes: list[str]) -> F:
    """Map func over the Cartesian product of product_axes.

    Like vmap, this function does not preserve the function signature and does not allow
    the function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched. Cannot have keyword-only
            arguments.
        product_axes (list): List with names of arguments over which we apply vmap.

    Returns:
        callable: A callable with the same arguments as func. See ``product_map`` for
            details.

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
