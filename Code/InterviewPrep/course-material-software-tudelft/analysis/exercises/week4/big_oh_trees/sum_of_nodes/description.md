For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

```python
def function_x(tree: BinaryTree) -> int:
    res = tree.val
    if tree.has_left():
        res += function_x(tree.left)
    if tree.has_right():
        res += function_x(tree.right)
    return res
```

1) Derive the run time equation of this code and explain all the terms.
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​

Note: the algorithm uses the following definition for the `BinaryTree` class:

```python
from __future__ import annotations

class BinaryTree:
    def __init__(self, val: int, left: BinaryTree = None, right: BinaryTree = None):
        self.val = val
        self.left = left
        self.right = right

    def has_left(self) -> bool:
        return self.left is not None

    def has_right(self) -> bool:
        return self.right is not None
```