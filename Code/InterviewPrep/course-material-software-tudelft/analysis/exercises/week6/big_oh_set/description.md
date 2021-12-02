For the code snippet below, work out what the worst-case run time complexity of `function_x` is in terms of big-Oh notation.

```python
from typing import Set, List

def function_x(xs: List[int]) -> Set[int]:
    ys = {x * 3 for x in xs}
    zs = {y // 2 for y in ys}
    return intersection(ys, zs)

def intersection(xs: Set[int], ys: Set[int]) -> Set[int]:
    res = set()
    for x in xs:
        if x in ys:
            res.add(x)
    return res

```

1) Derive the run time equation of this code and explain all the terms.

2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.
