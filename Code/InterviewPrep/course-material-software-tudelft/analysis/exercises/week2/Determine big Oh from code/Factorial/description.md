For the code snippet below, work out what the worst-case computational complexity is in terms of big-Oh notation.

```python
def factorial(n: int):
  if n == 0:
    return 1
  else:
    return n * factorial(n - 1)
```

1) Derive the recurrence equation for the runtime of this code and explain all the terms.

2) Derive the closed-form solution of this recurrence equation.

3) Simplify the closed-form equation​ and state the computational complexity of the `factorial` function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​
