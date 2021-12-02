For the code snippet below, express the time complexity in terms of a polynomial representation.
Simplify your polynomial representation to derive the time complexity in Big Oh notation.

**Note:** For this question you do not have to give the tightest upper bound.

**Hint:** You can use the fact that the time complexity of `fibonacci(x-1)` is upper bounded by `fibonacci(x)` in your derivation.

```python
def fibonacci(i: int) -> int:
    if i == 0:
        return 0
    if i == 1:
        return 1
    return fibonacci(i - 1) + fibonacci(i - 2)
```
