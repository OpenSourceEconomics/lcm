"""Small utility functions used troughout the project."""
import itertools
from collections.abc import Mapping


def map_dictionary(function, dictionary, by="keys"):
    """Apply a function to the keys or values of a flat or nested dictionary.

    Args:
        function (callable): function to be applied.
        dictionary (dict): Flat or nested dictionary to which the function is applied.
        by (str): Either "keys" or "values". Decides whether the function is applied to
            the dictionary keys or the dictionary values.

    Returns:
        dict: Modified copy of the dictionary.

    """
    if by == "values":
        if isinstance(dictionary, Mapping):
            return {k: map_dictionary(function, v, by) for k, v in dictionary.items()}
        else:
            return function(dictionary)

    elif by == "keys":
        if isinstance(dictionary, Mapping):
            return {
                function(k): map_dictionary(function, v, by)
                for k, v in dictionary.items()
            }
        else:
            return dictionary


def filter_dictionary(function, dictionary, by="keys"):
    """Filter a dictionary by conditions on keys or values.

    Args:
        function (callable): Function that takes one argument and returns True or False.
        dictionary (dict): Dictionary to be filtered.

    Returns:
        dict: Filtered dictionary

    Examples:
        >>> filter_dictionary(lambda x: "bla" in x, {"bla": 1, "blubb": 2})
        {'bla': 1}
        >>> filter_dictionary(lambda x: x <= 1, {"bla": 1, "blubb": 2}, by="values")
        {'bla': 1}

    """
    if by == "keys":
        keys_to_keep = set(filter(function, dictionary))
        out = {key: val for key, val in dictionary.items() if key in keys_to_keep}
    elif by == "values":
        out = {}
        for key, val in dictionary.items():
            if function(val):
                out[key] = val
    else:
        raise ValueError(f"by must be 'keys' or 'values', not {by}")

    return out


def update_dictionary(dictionary, other):
    """Create a copy of dictionary and update it with other.

    Args:
        dictionary (dict)
        other (dict)

    Returns:
        dict: The updated dictionary


    Examples:
        # make sure input is not changed
        >>> first = {"a": 1, "b": 2}
        >>> updated = update_dictionary(first, {"c": 3})
        >>> assert first == {"a": 1, "b": 2}

        # make sure update works
        >>> update_dictionary({"a": 1, "b": 2}, {"c": 3})
        {'a': 1, 'b': 2, 'c': 3}

    """
    return {**dictionary, **other}


def combine_dictionaries(dictionaries):
    """Combine a list of non-overlapping dictionaries into one.

    Args:
        dictionaries (list): List of dictionaries.

    Returns:
        dict: The combined dictionary.


    Examples:
        >>> combine_dictionaries([{"a": 1}, {"b": 2}])
        {'a': 1, 'b': 2}

    """
    if isinstance(dictionaries, dict):
        combined = dictionaries
    elif isinstance(dictionaries, list):
        if len(dictionaries) == 1:
            combined = dictionaries[0]
        else:
            key_sets = [set(d) for d in dictionaries]

            for first, second in itertools.combinations(key_sets, 2):
                intersection = first.intersection(second)
                if intersection:
                    raise ValueError(
                        f"The following keys occur more than once: {intersection}"
                    )

            combined = {}
            for d in dictionaries:
                combined = {**combined, **d}

    else:
        raise ValueError("'dictionaries' must be a dict or list of dicts.")

    return combined
