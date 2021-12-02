For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

```python
from typing import List


# Returns a sorted list in increasing order
def foo_sort(xs: List[int]) -> List[int]:
    res = []
    sz = len(xs)
    for it in range(sz):
        ys = xs.copy()
        for i in range(sz):
            for j in range(sz - 1):
                if ys[j] > ys[j + 1]:
                    ys[j], ys[j + 1] = ys[j + 1], ys[j]
        temp = ys[it]
        res.append(temp)
    return res
```

1) Derive the run time equation of this code and explain all the terms.

2) State the run time complexity of the `foo_sort` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
