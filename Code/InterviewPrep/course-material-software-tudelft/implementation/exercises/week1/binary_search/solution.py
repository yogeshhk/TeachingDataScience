from math import floor
from typing import List

from decorators import empty, remove


@empty
# Returns True if `item` can be found in the `items_list`, or False otherwise.
# `items_list` is a sorted list, so
def binary_search(items_list: List[int], item: int) -> bool:
    # return linear_time(items_list, item)
    # return logarithmic_time_iterative(items_list, item)
    return logarithmic_time_recursive(items_list, item)


@remove
# This implementation runs in O(n), which is too slow, so it will fail the big spec-test
def linear_time(items_list: List[int], item: int) -> bool:
    return item in items_list


@remove
def logarithmic_time_iterative(items_list: List[int], item: int) -> bool:
    first = 0
    last = len(items_list) - 1
    while first <= last:
        midpoint = floor((first + last) / 2)  # = (first + last) // 2
        if items_list[midpoint] == item:
            return True
        else:
            if item < items_list[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1
    return False


@remove
def logarithmic_time_recursive(items_list: List[int], item: int) -> bool:
    return logarithmic_time_recursive_helper(items_list, item, 0, len(items_list) - 1)


@remove
def logarithmic_time_recursive_helper(items_list: List[int], item: int, first: int, last: int) -> bool:
    if first > last:
        return False

    midpoint = (first + last) // 2  # = floor((first + last) / 2)
    if items_list[midpoint] == item:
        return True

    if item < items_list[midpoint]:
        return logarithmic_time_recursive_helper(items_list, item, first, midpoint - 1)
    else:
        return logarithmic_time_recursive_helper(items_list, item, midpoint + 1, last)
