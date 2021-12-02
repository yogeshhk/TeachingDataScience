```python
from typing import List


def bar(xs: List[int]):
    if len(xs) == 1:
        return xs[0]
    ys = []
    for i in xs[1:len(xs)]:
        ys.append(i)
    return xs[0] + bar(ys)
```