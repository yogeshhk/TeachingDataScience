For the code snippet below, work out the tightest worst-case run time complexity in terms of big-Oh notation.

```python
def method_y(n: int, m: int) -> int:
    res = m
    if n > 0:
        for i in range(n):
            res += method_y(n - 1, i)
    return res
```

1) Derive the (recursive) run time equation of the code and explain all the terms in your equations.

2) Derive the closed form of the run time equation.

3) State the run time complexity of the function in terms of Big Oh notation. You have to explain your answer, but a full proof is not needed.
