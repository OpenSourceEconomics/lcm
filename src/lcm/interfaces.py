from typing import NamedTuple

import numpy as np
import pandas as pd


class IndexerInfo(NamedTuple):
    """Information needed to work with an indexer array.

    In particular, this contains enough information to wrap an indexer array into a
    function that can be understood by dags.

    Attributes:
        axis_names (list): List of strings containing the names of the axes of the
            indexer array.
        name (str): The name of the indexer array. This will become an argument name
            of the function we need for dags.
        out_name (str): The name of the result of indexing into the indexer. This will
            become the name of the function we need for dags.

    """

    axis_names: list[str]
    name: str
    out_name: str


class GridSpec(NamedTuple):
    """Information needed to define or interpret a grid.

    Attributes:
        kind (str): Name of a grid type implemented in lcm.grids.
        specs (dict, np.ndarray): Specification of the grid. E.g. {"start": float,
            "stop": float, "n_points": int} for a linspace.

    """

    kind: str
    specs: dict | np.ndarray


class Space(NamedTuple):
    """Everything needed to evaluate a function on a space (e.g. state space).

    Attributes:
        sparse_vars (dict): Dictionary containing the names of sparse variables as keys
            and arrays with values of those variables as values. Together, the arrays
            define all feasible combinations of sparse variables.
        dense_vars (dict): Dictionary containing one dimensional grids of
            dense variables.

    """

    sparse_vars: dict
    dense_vars: dict


class SpaceInfo(NamedTuple):
    """Everything needed to work with the output of a function evaluated on a space.

    Attributes:
        axis_names (list): List with axis names of an array that contains function
            values for all elements in a space.
        lookup_info (dict): Dict that defines the possible labels of all discrete
            variables and their order.
        interpolation_info (dict): Dict that defines information on the grids of all
            continuous variables.
        indexer_infos (list): List of IndexerInfo objects.

    """

    axis_names: list[str]
    lookup_info: dict[str, list[str]]
    interpolation_info: dict[str, GridSpec]
    indexer_infos: list[IndexerInfo]


class Model(NamedTuple):
    """Internal representation of a user model.

    Attributes:
        grids (dict): Dictionary that maps names of model variables to grids of feasible
            values for that variable.
        gridspecs (dict): Dictionary that maps names of model variables to
            specifications from which grids of feasible values can be built.
        variable_info (pd.DataFrame): A table with information about all variables in
            the model. The index contains the name of a model variable. The columns are
            booleans that are True if the variable has the corresponding property. The
            columns are: is_state, is_choice, is_continuous, is_discrete, is_sparse,
            is_dense.
        functions (dict): Dictionary that maps names of functions to functions. The
            functions differ from the user functions in that they all except the
            filter functions take ``params`` as keyword argument. If the original
            function depended on model parameters, those are automatically extracted
            from ``params`` and passed to the original function. Otherwise, the
            ``params`` argument is simply ignored.
        function_info (pd.DataFrame): A table with information about all functions in
            the model. The index contains the name of a function. The columns are
            booleans that are True if the function has the corresponding property. The
            columns are: is_filter, is_constraint, is_next.
        params (dict): Dict of model parameters.
        shocks (dict): TODO
        n_periods (int): TODO

    """

    grids: dict
    gridspecs: dict
    variable_info: pd.DataFrame
    functions: dict
    function_info: pd.DataFrame
    params: dict
    # not really processed yet
    shocks: dict
    n_periods: int
