def all_as_kwargs(args, kwargs, arg_names):
    """Return kwargs dictionary containing all arguments.

    Args:
        args (tuple): Positional arguments.
        kwargs (dict): Keyword arguments.
        arg_names (list): Names of arguments.

    Returns:
        dict: A dictionary of all arguments.

    """
    return dict(zip(arg_names[: len(args)], args, strict=True)) | kwargs
