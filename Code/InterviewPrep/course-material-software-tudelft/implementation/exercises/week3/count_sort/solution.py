from typing import List

from decorators import empty


@empty
# Sorts in O(n+m) time, where m is the difference between the maximum and minimum value in xs
def count_sort(xs: List[int]) -> List[int]:
    min_val = min(xs)
    max_val = max(xs)

    counts = [0] * (max_val - min_val + 1)
    for x in xs:
        counts[x - min_val] += 1

    res = []
    for x in range(len(counts)):
        if counts[x] > 0:
            val = x + min_val
            res += [val] * counts[x]

    return res
