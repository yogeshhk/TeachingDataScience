from typing import List


def sum(a: List[int]) -> int:
    s = 0
    for i in a:
        s += i
    return s
