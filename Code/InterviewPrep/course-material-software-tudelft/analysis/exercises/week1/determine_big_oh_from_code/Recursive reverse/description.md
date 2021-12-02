For the code snippet below, express the time complexity of `reverse` in terms of a polynomial representation.
Simplify your polynomial representation to derive the time complexity in Big Oh notation.

```python
def reverse(xs: list[any]):
    reverse_helper(xs, 0, len(xs) - 1)


def reverse_helper(xs: list[any], l: int, h: int):
    if l < h:
        xs[l], xs[h] = xs[h], xs[l]
        reverse_helper(xs, l + 1, h - 1)
```
