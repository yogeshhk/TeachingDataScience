For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

This function will always be called with `x = 0`

```python
from typing import List

def selection_sort(xs: List[int], x: int):
    if x < len(xs) - 1:
        min_id = x
        for y in range(x + 1, len(xs)):
            if xs[y] < xs[min_id]:
                min_id = y
        xs[x], xs[min_id] = xs[min_id], xs[x]
        selection_sort(xs, x + 1)
```

1) Derive the recurrence equation for the runtime of this code and explain all the terms.

2) Derive the closed-form solution of this recurrence equation.

3) Simplify the given closed-form equation​ and state the computational complexity of the `selection_sort` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
