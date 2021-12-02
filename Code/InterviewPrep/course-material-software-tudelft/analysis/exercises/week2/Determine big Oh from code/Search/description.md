
For the code snippet below, work out what the worst-case computational complexity is in terms of big-Oh notation.
Take into account that \\(n\\), the amount of elements that is still to be searched, is `high - low`.

```python
from typing import List

def search(xs: List[int], low: int, high: int, x: int):
  if high < low:
    return 0
  if xs[low] == x:
    return 1
  if xs[high] == x:
    return 1
  return search(xs, low + 1, high - 1, x)
```

1) Derive the recurrence equation for the runtime of this code and explain all the terms.

2) Derive the closed-form solution of this recurrence equation, for the case where \\(n\\) is even.

The closed-form equation below provides a unified solution for any n (here, \\(\lfloor x \rfloor\\) is the _floor_ function).

$$ T(n) = \left\lfloor \frac{n}2 \right\rfloor \cdot c + c + b $$

3) Simplify the given closed-form equation​ and state the computational complexity of the `search` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
