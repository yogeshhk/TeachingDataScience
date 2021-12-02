from decorators import empty, remove
from .library import MWSTree, MWSTreeNode


@empty
# Tries to find the value in the multi way search tree,
# Returns true if the value is found, else false
def search(tree: MWSTree, val: int) -> bool:
    x = tree.root
    return recursive_helper(x, val)


@remove
def recursive_helper(n: MWSTreeNode, val: int) -> bool:
    if len(n.keys) == 0:
        return False
    idx = len(n.keys) - 1
    while idx >= 0 and val < n.keys[idx]:
        idx -= 1
    if n.keys[idx] == val:
        return True
    else:
        idx += 1
        if idx >= len(n.children):
            return False
        return recursive_helper(n.children[idx], val)
