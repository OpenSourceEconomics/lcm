import inspect

import networkx as nx


def concatenate_functions(functions, target):
    """Combine functions to one function that generates target.

    Functions can depend on the output of other functions as inputs, as long as the
    dependencies can be described by a directed acyclic graph (DAG).

    Functions that are not required to produce the target will simply be ignored.

    The arguments of the combined function are all arguments of relevant functions
    that are not themselves function names.

    Args:
        functions (dict or list): Dict or list of functions. If a list, the function
            name is inferred from the __name__ attribute of the entries. If a dict,
            the name of the function is set to the dictionary key.
        target (str): Name of the function that produces the target.

    Returns:
        function: A function that produces target when called with suitable arguments.

    """
    functions = _check_and_process_inputs(functions, target)
    raw_dag = _create_complete_dag(functions)
    dag = _limit_dag_to_targets_and_their_ancestors(raw_dag, target)
    signature = _get_signature(functions, dag)
    exec_info = _create_execution_info(functions, dag)
    concatenated = _create_concatenated_function(exec_info, signature)
    return concatenated


def get_ancestors(functions, target, include_target=False):
    """Build a DAG and extract all ancestors of target.

    Args:
        functions (dict or list): Dict or list of functions. If a list, the function
            name is inferred from the __name__ attribute of the entries. If a dict,
            the name of the function is set to the dictionary key.
        target (str): Name of the function that produces the target function.
        include_target (bool): Whether to include the target as its own ancestor.

    Returns:
        set: The ancestors

    """
    functions = _check_and_process_inputs(functions, target)
    raw_dag = _create_complete_dag(functions)
    dag = _limit_dag_to_targets_and_their_ancestors(raw_dag, target)

    ancestors = nx.ancestors(dag, target)
    if include_target:
        ancestors.add(target)
    return ancestors


def _check_and_process_inputs(functions, target):
    if isinstance(functions, list):
        functions = {func.__name__: func for func in functions}

    if not isinstance(target, str):
        raise ValueError(f"target must be a string, not {type(target)}")

    if target not in functions:
        # to-do: add typo suggestions via fuzzywuzzy, see estimagic or gettsim
        msg = f"The target '{target}' is not in functions."
        raise ValueError(msg)
    return functions


def _create_complete_dag(functions):
    """Create the complete DAG.

    This DAG is constructed from all functions and not pruned by specified root nodes or
    targets.

    Args:
        functions (dict): Dictionary containing functions to build the DAG.

    Returns:
        networkx.DiGraph: The complete DAG

    """
    functions_arguments_dict = {
        name: list(inspect.signature(function).parameters)
        for name, function in functions.items()
    }
    dag = nx.DiGraph(functions_arguments_dict).reverse()

    return dag


def _limit_dag_to_targets_and_their_ancestors(dag, target):
    """Limit DAG to targets and their ancestors.

    Args:
        dag (networkx.DiGraph): The complete DAG.
        target (str): Variable of interest.

    Returns:
        networkx.DiGraph: The pruned DAG.

    """
    used_nodes = {target}
    used_nodes = used_nodes | set(nx.ancestors(dag, target))

    all_nodes = set(dag.nodes)

    unused_nodes = all_nodes - used_nodes

    dag.remove_nodes_from(unused_nodes)

    return dag


def _get_signature(functions, dag):
    """Create the signature of the concatenated function.

    Args:
        functions (dict): Dictionary containing functions to build the DAG.
        dag (networkx.DiGraph): The complete DAG.

    Returns:
        inspect.Signature: The signature of the concatenated function.

    """
    function_names = set(functions)
    all_nodes = set(dag.nodes)
    arguments = sorted(all_nodes - function_names)

    parameter_objects = []
    for arg in arguments:
        parameter_objects.append(
            inspect.Parameter(name=arg, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )

    sig = inspect.Signature(parameters=parameter_objects)
    return sig


def _create_execution_info(functions, dag):
    """Create a dictionary with all information needed to execute relevant functions.

    Args:
        functions (dict): Dictionary containing functions to build the DAG.
        dag (networkx.DiGraph): The complete DAG.

    Returns:
        dict: Dictionary with functions and their arguments for each node in the dag.
            The functions are already in topological_sort order.

    """
    out = {}
    for node in nx.topological_sort(dag):
        if node in functions:
            info = {}
            info["func"] = functions[node]
            info["arguments"] = list(inspect.signature(functions[node]).parameters)

            out[node] = info
    return out


def _create_concatenated_function(execution_info, signature):
    """Create a concatenated function object with correct signature.

    Args:
        execution_info (dict): Dictionary with functions and their arguments for each
            node in the dag. The functions are already in topological_sort order.
        signature (inspect.Signature)): The signature of the concatenated function.

    Returns:
        callable: The concatenated function

    """
    parameters = sorted(signature.parameters)

    def concatenated(*args, **kwargs):
        results = {**dict(zip(parameters, args)), **kwargs}
        for name, info in execution_info.items():
            arguments = _dict_subset(results, info["arguments"])
            result = info["func"](**arguments)
            results[name] = result
        return result

    concatenated.__signature__ = signature

    return concatenated


def _dict_subset(dictionary, keys):
    """Reduce dictionary to keys."""
    return {k: dictionary[k] for k in keys}
