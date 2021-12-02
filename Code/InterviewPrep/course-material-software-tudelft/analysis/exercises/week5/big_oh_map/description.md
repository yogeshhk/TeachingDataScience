For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

```python
from typing import List, Dict


def combine(xs: List[Dict[int, int]]) -> Dict[int, int]:
    res = dict()

    combine_helper(xs, 0, len(xs) - 1, res)

    return res


def combine_helper(xs: List[Dict[int, int]], low: int, high: int, res: Dict[int, int]):
    if low == high:
        res.update(xs[low])
        return
    mid = low + (high - low) // 2
    combine_helper(xs, low, mid, res)
    combine_helper(xs, mid + 1, high, res)
```

Define \\(n = low - high + 1\\), where `low` and `high` are the integer parameters of `combineHelper`. Initially, \\(n\\) represents the size `xs`. In this list, every dictionary contains \\(m\\) entries (key-value pairs).
You can assume that `xs` contains an amount of maps that is a power of 2.

1) Derive the run time equation of `combine_helper`, you do not have to consider the `combine` method that calls `combine_helper`. Make sure to explain all the terms.
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
