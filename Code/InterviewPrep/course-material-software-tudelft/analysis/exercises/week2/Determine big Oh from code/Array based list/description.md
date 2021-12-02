For the code snippet below, work out what the worst-case computational complexity is in terms of big-Oh notation.
Take into account that \\(n\\), the amount of elements that is still to be searched, is `high - low`.
The function is always called with `low = 0`, `high = len(xs)`

Note: `xs` is a sorted (array-based) list.

```python
from typing import List

def binary_search(xs: List[int], n: int, low: int, high: int):
    if low - high == 1:
        return xs[low] == n
    mid_index = (high + low) // 2
    mid_num = xs[mid_index]
    if mid_num == n:
        return True
    elif mid_num > n:
        return binary_search(xs, n, low, mid_index)
    else:
        return binary_search(xs, n, mid_index + 1, high)
```

1) Derive the recurrence equation for the runtime of this code and explain all the terms.

2) Derive the closed-form solution of this recurrence equation.

3) Simplify the closed-form equation​ and state the computational complexity of the `binary_search` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
