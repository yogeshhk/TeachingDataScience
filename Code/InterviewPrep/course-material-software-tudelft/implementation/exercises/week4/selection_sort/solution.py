from typing import List
from decorators import empty


@empty
# A function receiving a list of integers as input
# and sorting them in increasing order using in-place selection sort.
def selection_sort(xs: List[int]):
    for x in range(len(xs)):
        min_index = x
        for y in range(x + 1, len(xs)):
            if xs[y] < xs[min_index]:
                min_index = y
        xs[x], xs[min_index] = xs[min_index], xs[x]
