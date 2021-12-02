from typing import List

from decorators import empty, remove


@empty
# Sorts the list in ascending order using quick sort
def sort(xs: List[int]) -> List[int]:
    if len(xs) < 2:
        return xs.copy()
    pivot = xs[0]  # Could also be xs[high] or a random index in the list
    remainder = xs[1:]
    left = [x for x in remainder if x < pivot]
    right = [x for x in remainder if x >= pivot]
    return sort(left) + [pivot] + sort(right)
