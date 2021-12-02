from .library import BinaryTree
from decorators import empty


@empty
# Returns true if the given value is in the tree
def search(val: int, tree: BinaryTree) -> bool:
    if tree.val == val:
        return True
    if val < tree.val and tree.has_left():
        return search(val, tree.left)
    if val > tree.val and tree.has_right():
        return search(val, tree.right)
    return False
