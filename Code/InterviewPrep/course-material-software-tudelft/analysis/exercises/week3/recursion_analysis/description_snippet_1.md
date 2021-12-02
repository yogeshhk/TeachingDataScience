```python
def foo(n: int):
    if n == 0:
        return 1
    return n * foo(n - 1)
```