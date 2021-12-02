from typing import List, Generator

from decorators import empty, remove


@empty
# Takes as input a number and a list.
# Returns the first (smallest) index of the number in the list.
# Returns None if the number cannot be found in the list.
def first_index_of(num: int, l: List[int]) -> int:
    for idx, val in enumerate(l):
        if val == num:
            return idx


@empty
# Takes as input a number and a list.
# Returns a generator with the indices of the number in the list.
# If the number cannot be found in the list, the generator will be empty.
def all_indices_of(num: int, l: List[int]) -> Generator[int, None, None]:
    return all_indices_of_iterative(num, l)
    # return all_indices_of_list_comprehension(num, l)


@remove
def all_indices_of_iterative(num: int, l: List[int]) -> Generator[int, None, None]:
    for idx, val in enumerate(l):
        if val == num:
            yield idx


@remove
def all_indices_of_list_comprehension(num: int, l: List[int]) -> Generator[int, None, None]:
    return (idx for idx, val in enumerate(l) if val == num)
