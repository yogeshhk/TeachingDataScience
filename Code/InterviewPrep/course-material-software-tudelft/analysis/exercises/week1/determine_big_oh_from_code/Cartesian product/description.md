For the code snippet below, express the time complexity in terms of a polynomial representation.
Simplify your polynomial representation to derive the time complexity in Big Oh notation.

Note: you can assume the two sets `xs` and `ys` have the same size.

```python
def cartesian_product(xs: set[T], ys: set[U]) -> set[tuple[T, U]]:
    res = set()
    for x in xs:
        for y in ys:
            res.add((x, y))
    return res
```
