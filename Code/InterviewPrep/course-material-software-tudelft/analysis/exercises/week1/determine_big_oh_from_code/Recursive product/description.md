For the code snippet below, express the time complexity in terms of a polynomial representation.
Simplify your polynomial representation to derive the time complexity in Big Oh notation.

```python
def multiply(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    if a < 0:
        return -b + multiply(a + 1, b)
    return b + multiply(a - 1, b)
```
