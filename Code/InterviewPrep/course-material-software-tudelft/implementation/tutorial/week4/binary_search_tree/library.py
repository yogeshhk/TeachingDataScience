from __future__ import annotations


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
