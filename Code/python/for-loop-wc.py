from typing import List


def foo(x: List[int]) -> int:
    s = 0
    for i in x:
        if i > 5:
            s += i * i
    return s
