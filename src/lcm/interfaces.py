from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from jax import Array

from lcm.grids import ContinuousGrid, DiscreteGrid, Grid
from lcm.typing import ParamsDict, ShockType


@dataclass(frozen=True)
class IndexerInfo:
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


@dataclass(frozen=True)
class Space:
    """Everything needed to evaluate a function on a space (e.g. state space).

    Attributes:
        sparse_vars (dict): Dictionary containing the names of sparse variables as keys
            and arrays with values of those variables as values. Together, the arrays
            define all feasible combinations of sparse variables.
        dense_vars (dict): Dictionary containing one dimensional grids of
            dense variables.

    """

    sparse_vars: dict[str, Array]
    dense_vars: dict[str, Array]


@dataclass(frozen=True)
class SpaceInfo:
    """Everything needed to work with the output of a function evaluated on a space.

    Attributes:
        axis_names: List with axis names of an array that contains function values for
            all elements in a space.
        lookup_info: Dict that defines the possible labels of all discrete variables and
            their order.
        interpolation_info: Dict that defines information on the grids of all continuous
            variables.
        indexer_infos: List of IndexerInfo objects.

    """

    axis_names: list[str]
    lookup_info: dict[str, DiscreteGrid]
    interpolation_info: dict[str, ContinuousGrid]
    indexer_infos: list[IndexerInfo]


@dataclass(frozen=True)
class InternalModel:
    """Internal representation of a user model.

    Attributes:
        grids: Dictionary that maps names of model variables to grids of feasible values
            for that variable.
        gridspecs: Dictionary that maps names of model variables to specifications from
            which grids of feasible values can be built.
        variable_info: A table with information about all variables in the model. The
            index contains the name of a model variable. The columns are booleans that
            are True if the variable has the corresponding property. The columns are:
            is_state, is_choice, is_continuous, is_discrete, is_sparse, columns are:
            is_state, is_choice, is_continuous, is_discrete, is_sparse, is_dense.
        functions: Dictionary that maps names of functions to functions. The functions
            differ from the user functions in that they all except the filter functions
            take ``params`` as keyword argument. If the original function depended on
            model parameters, those are automatically extracted from ``params`` and
            passed to the original function. Otherwise, the ``params`` argument is
            simply ignored.
        function_info: A table with information about all functions in the model. The
            index contains the name of a function. The columns are booleans that are
            True if the function has the corresponding property. The columns are:
            is_filter, is_constraint, is_next.
        params: Dict of model parameters.
        n_periods: Number of periods.
        random_utility_shocks: Type of random utility shocks.

    """

    grids: dict[str, Array]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    functions: dict[str, Callable]
    function_info: pd.DataFrame
    params: ParamsDict
    n_periods: int
    # Not properly processed yet
    random_utility_shocks: ShockType
