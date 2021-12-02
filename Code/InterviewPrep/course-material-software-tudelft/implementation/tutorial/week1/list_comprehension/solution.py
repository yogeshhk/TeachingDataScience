from typing import List

from decorators import empty


@empty
# Generates the list [0, 2, 6, 12, 20, 30, 42, 56, 72, 90]
def generate_list() -> List[int]:
    # return [] # This will fail
    # return [0,2,6,12,20,30,42,56,72,90] # But this is no list comprehension
    # return [ x * (x + 1) for x in range(0,10)]
    return [x * (x + 1) for x in range(10)]
