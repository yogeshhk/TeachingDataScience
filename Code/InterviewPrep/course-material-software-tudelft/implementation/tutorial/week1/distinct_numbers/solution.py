from typing import List

from decorators import empty, remove


@empty
# Returns true iff all numbers in xs are distinct
def is_distinct(xs: List[int]) -> bool:
    return is_distinct_linear(xs)
    # return is_distinct_quadratic(xs)


@remove
def dict_solution(xs: List[int]) -> bool:
    d = dict()
    for item in xs:
        if item in d:
            d[item] += 1  # could also just be return False
        else:
            d[item] = 1

    for key in d:
        if d[key] > 1:
            return False
    return True


@remove
# This solution runs in linear time
def is_distinct_linear(xs: List[int]) -> bool:
    return len(xs) == len(set(xs))


@remove
# This solution runs in quadratic time
def is_distinct_quadratic(xs: List[int]) -> bool:
    for i in range(len(xs)):
        for j in range(len(xs)):
            if xs[i] == xs[j] and i != j:
                return False
    return True
