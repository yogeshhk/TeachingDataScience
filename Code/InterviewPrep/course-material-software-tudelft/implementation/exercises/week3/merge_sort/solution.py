from typing import List
from decorators import empty, remove


@empty
# A function receiving a list of integers as input
# and sorting them in increasing order using merge sort
def sort(items_list: List[int]) -> List[int]:
    if len(items_list) < 2:
        return items_list.copy()
    middle = len(items_list) // 2
    left = sort(items_list[:middle])
    right = sort(items_list[middle:])
    return merge(left, right)


@remove
# Helper function for merging two sorted lists
def merge(left: List[int], right: List[int]) -> List[int]:
    i = j = 0
    res = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    return res + left[i:] + right[j:]
