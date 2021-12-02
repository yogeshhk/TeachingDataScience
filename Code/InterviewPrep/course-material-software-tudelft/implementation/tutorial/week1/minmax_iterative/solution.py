from typing import List

from decorators import empty


@empty
# Returns the minimum and maximum value of the given list as a tuple (min, max)
def minmax(xs: List[int]) -> (int, int):
    # return sorting_them_first(xs)
    return iteration(xs)


@remove
# This does not really meet the requirements as this does not iterate over all items once.
def sorting_them_first(xs: List[int]) -> (int, int):
    xs.sort()  # This requires O(n \log n) time!
    return xs[0], xs[-1]


@remove
def iteration(xs: List[int]) -> (int, int):
    mi = xs[0]  # Alternatively: mi = ma = xs[0]
    ma = xs[0]
    for x in xs:
        if x < mi:
            mi = x
        if x > ma:
            ma = x
    return mi, ma
