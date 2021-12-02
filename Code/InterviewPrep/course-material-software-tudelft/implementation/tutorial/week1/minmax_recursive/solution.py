from typing import List

from decorators import empty, remove


@empty
# Returns the minimum and maximum value of the given list as a tuple (min, max)
def minmax(xs: List[int]) -> (int, int):
    return minmax_helper(xs, 0)  # Regular linear implementation
    # return minmax_tail_helper(xs, 0, xs[0], xs[0])  # Tail recursive linear implementation
    # return minmax_quadratic(xs)  # Quadratic implementation


@remove
# Helper method to recursively find min and max value in linear time
def minmax_helper(xs: List[int], index: int) -> (int, int):
    if index == len(xs) - 1:  # When at the last element,
        return xs[-1], xs[-1]  # Return that element as min and max.
    min, max = minmax_helper(xs, index + 1)  # Get min and max from the rest of the list.
    if xs[index] < min:  # Change min if current value is smaller.
        min = xs[index]
    if xs[index] > max:  # Change max if current value is smaller.
        max = xs[index]
    return min, max


@remove
# Runs in quadratic time due to using list slicing
def minmax_quadratic(xs: List[int]) -> (int, int):
    if len(xs) == 1:
        return xs[0], xs[0]
    min, max = minmax_quadratic(xs[1:])  # This creates a new list, taking linear time
    if xs[0] < min:
        min = xs[0]
    if xs[0] > max:
        max = xs[0]
    return min, max


@remove
# Helper method for tail recursive implementation
def minmax_tail_helper(xs: List[int], index: int, min: int, max: int) -> (int, int):
    if index >= len(xs):
        return min, max
    cur = xs[index]
    if cur < min:
        min = cur
    elif cur > max:
        max = cur
    return minmax_tail_helper(xs, index + 1, min, max)  # Traverse the rest of the list with updated min and max
