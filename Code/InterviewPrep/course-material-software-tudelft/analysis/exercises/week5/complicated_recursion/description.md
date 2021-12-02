For the code snippet below, work out the tightest worst-case run time complexity in terms of big-Oh notation.

```python
from typing import List, Tuple


def method_x(xs: List[int]) -> List[Tuple[int, int]]:
    if len(xs) < 2:
        return list()
    a = [(xs[i], xs[-1]) for i in range(len(xs) - 1)]
    b = method_x(xs[:-1])
    return a + b
```

1) Derive the run time equation of the code and explain all the terms in your equations.

2) Derive the closed form of the run time equation.

3) State the run time complexity of the function in terms of Big Oh notation. You have to explain your answer, but a full proof is not needed.
