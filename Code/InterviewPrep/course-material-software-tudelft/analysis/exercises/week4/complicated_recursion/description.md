For the code snippet below representing a sorting algorithm, work out the worst-case run time complexity in terms of big-Oh notation.

```python
import math

def method_y(n: int) -> int:
    if n == 0:
        return 1
    a = method_y(math.ceil(n / 5))
    b = method_y(math.floor(n / 5))
    res = 0
    for i in range(math.sqrt(n)):
        res += math.floor(a / i)
    return res * b
```

1) Derive the (recursive) run time equation of the code and explain all the terms in your equations.

2) State the run time complexity of the function in terms of Big Oh notation. You have to explain your answer, but a full proof is not needed.
