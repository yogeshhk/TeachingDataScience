For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

```python
from typing import List

def function_z(xs: List[int]) -> List[int]:
    hp = Heap()
    for x in xs:
        hp.add(x)
    return [hp.remove_min() for _ in range(len(xs))]
```

1) Derive the run time equation of this code and explain all the terms.
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.

**Note**: assume the `Heap` class implements a min-heap.
