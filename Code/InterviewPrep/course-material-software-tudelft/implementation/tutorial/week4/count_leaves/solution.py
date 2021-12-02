from .library import BinaryTree
from decorators import empty, remove


@empty
# Returns the amount of leaf nodes in the given binary tree
def count_leaves(tree: BinaryTree) -> int:
    # return count_leaves_1(tree)
    return count_leaves_2(tree)


@remove
# Uses has_left and has_right to determine when to stop
def count_leaves_1(tree: BinaryTree) -> int:
    if not tree.has_left() and not tree.has_right():
        return 1
    res = 0
    if tree.has_left():
        res += count_leaves_1(tree.left)
    if tree.has_right():
        res += count_leaves_1(tree.right)
    return res


@remove
# Alternative solution without has_left and has_right, return 0 when tree is None
def count_leaves_2(tree: BinaryTree) -> int:
    if tree is None:
        return 0
    if tree.has_left() or tree.has_right():
        return count_leaves_2(tree.left) + count_leaves_2(tree.right)
    return 1
