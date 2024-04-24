import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable)


# ======================================================================================
# Decorators
# ======================================================================================


def allow_only_kwargs(func: F) -> F:
    """Restrict a function to be called with *only* keyword arguments.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: A Callable with the same arguments as func (but with the additional
            restriction that it is *only* callable with keyword arguments).

    """
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Get names of keyword-only arguments
    kw_only_parameters = [
        p.name for p in parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]

    # Create new signature without positional-only arguments
    new_parameters = [
        (
            p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            if p.kind == inspect.Parameter.POSITIONAL_ONLY
            else p
        )
        for p in parameters.values()
    ]
    new_signature = signature.replace(parameters=new_parameters)

    @functools.wraps(func)
    def allow_only_kwargs_wrapper(**kwargs):
        # Check if the total number of arguments matches the function signature
        if extra := set(kwargs).difference(parameters):
            raise ValueError(f"Unknown arguments provided: {extra}")

        if missing := set(parameters).difference(kwargs):
            raise ValueError(f"Missing arguments: {missing}")

        # Retrieve keyword-only arguments
        kw_only_kwargs = {k: kwargs[k] for k in kw_only_parameters}

        # Get kwargs that must be converted to positional arguments
        pos_kwargs = {k: v for k, v in kwargs.items() if k not in kw_only_parameters}

        # Collect all positional arguments in correct order
        positional = convert_kwargs_to_args(pos_kwargs, list(parameters))

        return func(*positional, **kw_only_kwargs)

    allow_only_kwargs_wrapper.__signature__ = new_signature
    return allow_only_kwargs_wrapper


def allow_kwargs(func: F) -> F:
    """Allow a function to be called with keyword arguments.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: A Callable with the same arguments as func (but with the additional
            possibility to call it with keyword arguments).

    """
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Get names of keyword-only arguments
    kw_only_parameters = [
        p.name for p in parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]

    # Create new signature without positional-only arguments
    new_parameters = [
        (
            p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            if p.kind == inspect.Parameter.POSITIONAL_ONLY
            else p
        )
        for p in parameters.values()
    ]
    new_signature = signature.replace(parameters=new_parameters)

    @functools.wraps(func)
    def allow_kwargs_wrapper(*args, **kwargs):
        # Check if the total number of arguments matches the function signature
        if len(args) + len(kwargs) != len(parameters):
            raise ValueError(_error_message(args, kwargs, parameters))

        # Retrieve keyword-only arguments
        kw_only_kwargs = {k: kwargs[k] for k in kw_only_parameters}

        # Get kwargs that must be converted to positional arguments
        pos_kwargs = {k: v for k, v in kwargs.items() if k not in kw_only_parameters}

        # Collect all positional arguments in correct order
        positional = list(args) + convert_kwargs_to_args(pos_kwargs, list(parameters))

        return func(*positional, **kw_only_kwargs)

    allow_kwargs_wrapper.__signature__ = new_signature
    return allow_kwargs_wrapper


def allow_args(func: F) -> F:
    """Allow a function to be called with positional arguments.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: A Callable with the same arguments as func (but with the additional
            possibility to call it with positional arguments).

    """
    signature = inspect.signature(func)
    parameters = signature.parameters

    # Count the number of positional-only arguments
    n_positional_only_parameters = len(
        [p for p in parameters.values() if p.kind == inspect.Parameter.POSITIONAL_ONLY],
    )

    # Create new signature without keyword-only arguments
    new_parameters = [
        (
            p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            else p
        )
        for p in parameters.values()
    ]
    new_signature = signature.replace(parameters=new_parameters)

    @functools.wraps(func)
    def allow_args_wrapper(*args, **kwargs):
        # Check if the total number of arguments matches the function signature
        if len(args) + len(kwargs) != len(parameters):
            raise ValueError(_error_message(args, kwargs, parameters))

        # Convert all arguments to positional arguments in correct order
        positional = list(args) + convert_kwargs_to_args(kwargs, list(parameters))

        # Extract positional-only arguments
        positional_only = positional[:n_positional_only_parameters]

        # Create kwargs dictionary with remaining arguments
        kwargs_names = list(parameters)[n_positional_only_parameters:]
        kwargs = dict(
            zip(
                kwargs_names,
                positional[n_positional_only_parameters:],
                strict=True,
            ),
        )

        return func(*positional_only, **kwargs)

    allow_args_wrapper.__signature__ = new_signature
    return allow_args_wrapper


# ======================================================================================
# Auxiliary functions
# ======================================================================================


def get_union_of_arguments(list_of_functions: list[Callable]) -> set[str]:
    """Return the union of arguments of a list of functions.

    Args:
        list_of_functions (list): A list of functions.

    Returns:
        set: The union of arguments of all functions in list_of_functions.

    """
    arguments = [inspect.signature(f).parameters for f in list_of_functions]
    return set().union(*arguments)


def all_as_kwargs(
    args: tuple[Any],
    kwargs: dict[str, Any],
    arg_names: list[str],
) -> dict[str, Any]:
    """Return kwargs dictionary containing all arguments.

    Args:
        args (tuple): Positional arguments.
        kwargs (dict): Keyword arguments.
        arg_names (list): Names of arguments.

    Returns:
        dict: A dictionary of all arguments.

    """
    return dict(zip(arg_names[: len(args)], args, strict=True)) | kwargs


def all_as_args(
    args: tuple[Any],
    kwargs: dict[str, Any],
    arg_names: list[str],
) -> tuple[Any]:
    """Return args tuple containing all arguments.

    Args:
        args (tuple): Positional arguments.
        kwargs (dict): Keyword arguments.
        arg_names (list): Names of arguments.

    Returns:
        tuple: A tuple of all arguments.

    """
    return args + tuple(convert_kwargs_to_args(kwargs, arg_names))


def convert_kwargs_to_args(kwargs: dict[str, Any], parameters: list[str]) -> list[Any]:
    """Convert kwargs to args in the order of parameters.

    Args:
        kwargs (dict): Keyword arguments.
        parameters (list): List of parameter names in the order they should be.

    Returns:
        list: List of arguments in the order of parameters.

    """
    sorted_kwargs = dict(sorted(kwargs.items(), key=lambda kw: parameters.index(kw[0])))
    return list(sorted_kwargs.values())


def _error_message(
    args: tuple[Any],
    kwargs: dict[str, Any],
    parameters: list[str],
) -> str:
    too_many = len(args) + len(kwargs) > len(parameters)
    return "Too many arguments provided." if too_many else "Not all arguments provided."
