For the code snippet below representing a sorting algorithm, work out the worst-case run time complexity in terms of big-Oh notation.

```python
from typing import List


def method_x(xs: List[int], n: int):
    if n == 0:
        return
    for i in range(n - 1):
        if xs[i + 1] > xs[i]:
            xs[i], xs[i + 1] = xs[i + 1], xs[i]
    return method_x(xs, n - 1)
```

1) Derive the run time equation of the code and explain all the terms in your equations.

2) State the run time complexity of the function in terms of Big Oh notation. You have to explain your answer, but full 
proof is not needed.​

