import functools
import inspect

from jax import vmap


def state_space_map(func, simple_variables, complex_variables):
    """Apply vmap such that a function can be evaluated on the state space.

    This is achieved by applying a product map for all simple_variables and a
    vmap for the complex variables.

    In contrast to vmap, product_map preserves the function signature and allows the
    function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        simple_variables (list): Names of the simple variables in the
            state_choice_space.
        complex_variables (list): Names of the complex variables in the
            state_choice_space.
    """
    if not set(simple_variables).isdisjoint(complex_variables):
        raise ValueError("Simple and complex variables overlap.")

    _all_variables = simple_variables + complex_variables
    if len(_all_variables) != len(set(_all_variables)):
        raise ValueError("Same argument provided more than once.")

    signature = inspect.signature(func)
    parameters = list(signature.parameters)

    positions = [parameters.index(cv) for cv in complex_variables]
    in_axes = []
    for i in range(len(parameters)):
        if i in positions:
            in_axes.append(0)
        else:
            in_axes.append(None)

    vmapped = vmap(func, in_axes=in_axes)
    vmapped = _product_map(vmapped, simple_variables)

    vmapped.__signature__ = signature
    vmapped_with_kwargs = allow_kwargs(vmapped)

    return vmapped_with_kwargs


def product_map(func, product_axes):
    """Apply vmap such that func is applied to cartesian product of product_axes.

    This requires an iterative application of vmap.

    In contrast to vmap, product_map preserves the function signature and allows the
    function to be called with keyword arguments.

    Args:
        func (callable): The function to be dispatched.
        product_axes (list): List with names of arguments that over which the cartesian
            product should be formed.
    """
    if len(product_axes) != len(set(product_axes)):
        raise ValueError("Same argument provided more than once.")

    signature = inspect.signature(func)
    vmapped = _product_map(func, product_axes)
    vmapped.__signature__ = signature
    vmapped_with_kwargs = allow_kwargs(vmapped)

    return vmapped_with_kwargs


def _product_map(func, product_axes):
    """Do actual product map without signature changes."""
    signature = inspect.signature(func)
    parameters = list(signature.parameters)
    if product_axes is None:
        product_axes = parameters

    positions = [parameters.index(ax) for ax in product_axes]

    vmap_specs = []
    for pos in reversed(positions):
        spec = [None] * len(parameters)
        spec[pos] = 0
        vmap_specs.append(spec)

    vmapped = func
    for spec in vmap_specs:
        vmapped = vmap(vmapped, in_axes=spec)

    return vmapped


def allow_kwargs(func):
    @functools.wraps(func)
    def allow_kwargs_wrapper(*args, **kwargs):
        parameters = list(inspect.signature(func).parameters)

        positional = list(args) if args is not None else []

        kwargs = {} if args is None else kwargs
        if len(args) + len(kwargs) != len(parameters):
            raise ValueError("Not enough or too many arguments provided.")

        positional += convert_kwargs_to_args(kwargs, parameters)

        return func(*positional)

    return allow_kwargs_wrapper


def convert_kwargs_to_args(kwargs, parameters):
    sorted_kwargs = dict(sorted(kwargs.items(), key=lambda kw: parameters.index(kw[0])))
    args = list(sorted_kwargs.values())
    return args
