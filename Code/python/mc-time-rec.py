from typing import List


def recfunc(xs: List[int]) -> List[int]:
    if len(xs) < 2:
        return xs
    a = list()
    b = list()
    for x in range(len(xs)):
        if x < len(xs) // 2:
            a.insert(0, x)
        else:
            b.insert(0, x)
    return recfunc(a) + recfunc(b)
