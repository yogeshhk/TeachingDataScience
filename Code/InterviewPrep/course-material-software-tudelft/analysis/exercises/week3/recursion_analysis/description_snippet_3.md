```python
from typing import List


def foobar(xs: List[int]):
    if len(xs) <= 1:
        return xs
    i = len(xs) // 2
    ys = xs[0:i]
    zs = xs[i: len(xs)]
    return foobar(zs) + foobar(ys)
```