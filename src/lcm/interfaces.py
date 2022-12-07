from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Union

import numpy as np


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

    axis_names: List[str]
    name: str
    out_name: str


class Grid(NamedTuple):
    """Information needed to define or interpret a grid.

    Attributes:
        kind (str): Name of a grid type implemented in lcm.grids.
        specs (dict, np.ndarray): Specification of the grid. E.g. {"start": float,
            "stop": float, "n_points": int} for a linspace.

    """

    kind: str
    specs: Union[dict, np.ndarray]


class Space(NamedTuple):
    """Everything needed to evaluate a function on a space (e.g. state space).

    Attributes:
        sparse_vars (dict): Dictionary containing the names of sparse variables as keys
            and arrays with values of those variables as values. Together, the arrays
            define all feasible combinations of sparse variables.
        dense_vars (dict): Dictionary containing one dimensional grids of
            dense variables.

    """

    sparse_vars: Dict
    dense_vars: Dict


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

    axis_names: List[str]
    lookup_info: Dict[str, List[str]]
    interpolation_info: Dict[str, Grid]
    indexer_infos: List[IndexerInfo]
