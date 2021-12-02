For the code snippet below, work out what the worst-case computational complexity is in terms of big-Oh notation.
Take into account that \\(n\\), the amount of elements that is still to be searched, is `high - low`.
```python
def sum_of_sums(tot: int, n: int):
    if n == 0:
        return tot
    for x in range(1, n + 1):
        tot += x
    return sum_of_sums(tot, n - 1)
```

1) Derive the recurrence equation for the runtime of this code and explain all the terms.

2) Derive the closed-form solution of this recurrence equation.

3) Simplify the closed-form equation​ and state the computational complexity of the `sum_of_sums` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
