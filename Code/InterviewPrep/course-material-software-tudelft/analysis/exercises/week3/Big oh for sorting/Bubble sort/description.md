For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

This function will always be called with `x = len(xs)`

```python
from typing import List

def bubble_sort(xs: List[int]):
    for x in range(len(xs), 1, -1):
        for y in range(0, x - 1):
            if xs[y] > xs[y + 1]:
                xs[y], xs[y + 1] = xs[y + 1], xs[y]
```

1) Derive the run time equation of this code and explain all the terms.

2) State the run time complexity of the `bubble_sort` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
