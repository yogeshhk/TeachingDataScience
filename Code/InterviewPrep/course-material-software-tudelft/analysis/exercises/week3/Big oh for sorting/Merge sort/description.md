For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

```python
from typing import List

def merge_sort(xs: List[int]) -> List[int]:
    if len(xs) < 2:
        return xs.copy()
    mid = len(xs) // 2
    left = xs[:mid]
    right = xs[mid:]
    left = merge_sort(left)
    right = merge_sort(right)
    return merge(left, right)
    
def merge(left: List[int], right: List[int]) -> List[int]:
    i = j = 0
    res = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    return res + left[i:] + right[j:]
```

1) Derive the recurrence equation for the runtime of this code and explain all the terms.

2) State the run time complexity of the `merge_sort` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​

Note: you can use the master method to solve 2).
