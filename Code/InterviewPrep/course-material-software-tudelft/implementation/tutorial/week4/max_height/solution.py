from .library import BinaryTree
from decorators import empty


@empty
# Returns the maximum height of the given binary tree
def max_height(tree: BinaryTree) -> int:
    if tree is None:
        return 0
    return max(max_height(tree.right), max_height(tree.left)) + 1
