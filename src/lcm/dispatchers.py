import inspect
from collections.abc import Callable
from typing import Literal, TypeVar, cast

from jax import Array, vmap

from lcm.functools import allow_args, allow_only_kwargs
from lcm.utils import find_duplicates

F = TypeVar("F", bound=Callable[..., Array | tuple[Array, Array]])


def spacemap(
    func: F,
    product_vars: tuple[str, ...],
    combination_vars: tuple[str, ...],
) -> F:
    """Apply vmap such that func can be evaluated on product and combination variables.

    Product variables are used to create a Cartesian product of possible values. I.e.,
    for each product variable, we create a new leading dimension in the output object,
    with the size of the dimension being the number of possible values in the grid. The
    i-th entries of the combination variables, correspond to one valid combination. For
    the combination variables, a single dimension is thus added to the output object,
    with the size of the dimension being the number of possible combinations. This means
    that all combination variables must have the same size (e.g., in the simulation the
    states act as combination variables, and their size equals the number of
    simulations).

    spacemap preserves the function signature and allows the function to be called with
    keyword arguments.

    Args:
        func: The function to be dispatched.
        product_vars: Names of the product variables, i.e. those that are stored as
            arrays of possible values in the grid, over which we create a Cartesian
            product.
        combination_vars: Names of the combination variables, i.e. those that are
            stored as arrays of possible combinations.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns a jax.Array or pytree of arrays. If `func` returns a
        scalar, the dispatched function returns a jax.Array with k + 1 dimensions, where
        k is the length of `product_vars` and the additional dimension corresponds to
        the `combination_vars`. The order of the dimensions is determined by the order
        of `product_vars`. If the output of `func` is a jax pytree, the usual jax
        behavior applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    if duplicates := find_duplicates(product_vars, combination_vars):
        msg = (
            "Same argument provided more than once in product variables or combination "
            f"variables, or is present in both: {duplicates}"
        )
        raise ValueError(msg)

    func_callable_with_args = allow_args(func)

    vmapped = _base_productmap(func_callable_with_args, product_vars)

    if combination_vars:
        vmapped = vmap_1d(
            vmapped, variables=combination_vars, callable_with="only_args"
        )

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = inspect.signature(func_callable_with_args)  # type: ignore[attr-defined]

    return cast(F, allow_only_kwargs(vmapped))


def vmap_1d(
    func: F,
    variables: tuple[str, ...],
    *,
    callable_with: Literal["only_args", "only_kwargs"] = "only_kwargs",
) -> F:
    """Apply vmap such that func is mapped over the specified variables.

    In contrast to a general vmap call, vmap_1d vectorizes along the leading axis of all
    of the requested variables simultaneously. Moreover, it preserves the function
    signature and allows the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched.
        variables: Tuple with names of arguments that over which we map.
        callable_with: Whether to apply the allow_kwargs decorator to the dispatched
            function. If "only_args", the returned function can only be called with
            positional arguments. If "only_kwargs", the returned function can only be
            called with keyword arguments.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns a jax.Array or pytree of arrays. If `func`
        returns a scalar, the dispatched function returns a jax.Array with 1
        jax.Array with 1 dimension and length k, where k is the length of one of
        the mapped inputs in `variables`. The order of the dimensions is determined by
        the order of `variables` which can be different to the order of `funcs`
        arguments. If the output of `func` is a jax pytree, the usual jax behavior
        applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    if duplicates := find_duplicates(variables):
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

    return cast(F, out)


def productmap(func: F, variables: tuple[str, ...]) -> F:
    """Apply vmap such that func is evaluated on the Cartesian product of variables.

    This is achieved by an iterative application of vmap.

    In contrast to _base_productmap, productmap preserves the function signature and
    allows the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched.
        variables: Tuple with names of arguments that over which the Cartesian product
            should be formed.

    Returns:
        A callable with the same arguments as func (but with an additional leading
        dimension) that returns a jax.Array or pytree of arrays. If `func`
        returns a scalar, the dispatched function returns a jax.Array with k
        dimensions, where k is the length of `variables`. The order of the dimensions
        is determined by the order of `variables` which can be different to the order
        of `funcs` arguments. If the output of `func` is a jax pytree, the usual jax
        behavior applies, i.e. the leading dimensions of all arrays in the pytree are as
        described above but there might be additional dimensions.

    """
    if duplicates := find_duplicates(variables):
        raise ValueError(
            f"Same argument provided more than once in variables: {duplicates}",
        )

    func_callable_with_args = allow_args(func)

    vmapped = _base_productmap(func_callable_with_args, variables)

    # This raises a mypy error but is perfectly fine to do. See
    # https://github.com/python/mypy/issues/12472
    vmapped.__signature__ = inspect.signature(func_callable_with_args)  # type: ignore[attr-defined]

    return cast(F, allow_only_kwargs(vmapped))


def _base_productmap(func: F, product_axes: tuple[str, ...]) -> F:
    """Map func over the Cartesian product of product_axes.

    Like vmap, this function does not preserve the function signature and does not allow
    the function to be called with keyword arguments.

    Args:
        func: The function to be dispatched. Cannot have keyword-only arguments.
        product_axes: Tuple with names of arguments over which we apply vmap.

    Returns:
        A callable with the same arguments as func. See `product_map` for details.

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
