from typing import List

import decorators


def list_comprehension_function1(a: int, b: int) -> List[int]:
    return [x ** 2 for x in range(a, b + 1)]


@decorators.empty
# In this function you should implement the iterative way
# of constructing the list that `list_comprehension_function1`
# is creating.
def iterative_function1(a: int, b: int) -> List[int]:
    result = []
    for num in range(a, b + 1):
        result.append(num ** 2)
    return result


def iterative_function2(a: int, b: int) -> List[int]:
    result = []
    for num in range(a, b + 1):
        result.append(num * 3)
    return result


@decorators.empty
# Now you are given the iterative way of creating a list in `iterative_function2`
# and we ask you to come up with a list comprehension that constructs the same list.
def list_comprehension_function2(a: int, b: int) -> List[int]:
    return [num * 3 for num in range(a, b + 1)]
