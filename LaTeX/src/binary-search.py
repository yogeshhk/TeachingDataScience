from typing import List


def bin_s(L: List[int], v: int) -> bool:
    if len(L) == 0:
        return False

    m = len(L) // 2  # Rounds down
    if L[m] == v:
        return True

    if L[m] > v:
        return bin_s(L[:m], v)
    else:
        return bin_s(L[m + 1:], v)
