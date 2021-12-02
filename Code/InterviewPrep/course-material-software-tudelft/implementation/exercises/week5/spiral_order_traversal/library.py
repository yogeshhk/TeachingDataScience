from __future__ import annotations

from collections import deque


class BinaryTree:
    def __init__(self, val: int, left: BinaryTree = None, right: BinaryTree = None):
        # Note that left and right are optional parameters: calling BinaryTree(1) results in a tree without children.
        # As you can see in the has_left and has_right functions, the value of left or right is None when
        # there is no child in this location.

        self.val = val
        self.left = left
        self.right = right

    def has_left(self) -> bool:
        return self.left is not None

    def has_right(self) -> bool:
        return self.right is not None


class Queue:
    def __init__(self):
        self.q = deque()

    def enqueue(self, item):
        self.q.append(item)

    def dequeue(self):
        return self.q.popleft()

    def first(self):
        return self.q[0]

    def __len__(self):
        return len(self.q)

    def is_empty(self):
        return len(self.q) == 0
