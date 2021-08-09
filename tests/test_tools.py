from lcm.tools import map_dictionary


def test_map_nested_dictionary_by_values():
    d = {"a": 2, "b": {"c": 3, "d": 4}}
    calculated = map_dictionary(lambda x: x ** 2, d, "values")
    expected = {"a": 4, "b": {"c": 9, "d": 16}}
    assert calculated == expected


def test_map_nested_dictionary_by_keys():
    d = {"a": 2, "b": {"c": 3, "d": 4}}
    calculated = map_dictionary(lambda x: x * 2, d, "keys")
    expected = {"aa": 2, "bb": {"cc": 3, "dd": 4}}
    assert calculated == expected
