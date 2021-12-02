For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

```python
def foo(tree: BinarySearchTree) -> int:
    res = 0
    if not (tree.has_right() or tree.has_left()):
        return res
    if tree.has_right() and tree.has_left():
        res += 1
    return res + foo(tree.right) + foo(tree.left)
```

1) Derive the run time equation of this code and explain all the terms.
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​

The algorithm uses the following definition for the `BinarySearchTree` class:

```python
from __future__ import annotations


class BinarySearchTree:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None

    def has_left(self) -> bool:
        return self.left is not None

    def has_right(self) -> bool:
        return self.right is not None

    def add(self, val: int):
        if val <= self.val:
            if self.has_left():
                self.left.add(val)
            else:
                self.left = BinarySearchTree(val)
        else:
            if self.has_right():
                self.right.add(val)
            else:
                self.right = BinarySearchTree(val)

```