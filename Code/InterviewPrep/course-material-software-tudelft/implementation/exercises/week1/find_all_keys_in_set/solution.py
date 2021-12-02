from typing import Dict, Set

from decorators import empty, remove


@empty
# Takes as input a number and a dictionary,
# Returns a set with all the keys, for which the number is contained in its set.
def find_keys(num: int, my_dict: Dict[str, Set[int]]) -> Set[str]:
    return find_keys_iterative(num, my_dict)
    # return find_keys_list_comprehension(num, my_dict)


@remove
def find_keys_list_comprehension(num: int, my_dict: Dict[str, Set[int]]) -> Set[str]:
    return set(idx for idx, value in my_dict.items() if num in value)


@remove
def find_keys_iterative(num: int, my_dict: Dict[str, Set[int]]) -> Set[str]:
    result = set()
    for key in my_dict:
        if num in my_dict[key]:
            result.add(key)
    return result
