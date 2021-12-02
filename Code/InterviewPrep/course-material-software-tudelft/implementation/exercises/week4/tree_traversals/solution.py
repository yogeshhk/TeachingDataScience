from typing import List

from decorators import empty, remove
from .library import BinaryTree


@empty
# Returns a list of the values with a preorder traversal
def preorder_traversal(tree: BinaryTree) -> List[int]:
    # Returns a list with the results
    return preorder_get_all_nodes(tree)
    # Takes a list as parameter
    # res = []
    # preorder_get_all_nodes_extra_list(tree, res)
    # return res


@remove
def preorder_get_all_nodes_extra_list(tree: BinaryTree, res: List[int]):
    res.append(tree.val)
    if tree.has_left():
        preorder_get_all_nodes_extra_list(tree.left, res)
    if tree.has_right():
        preorder_get_all_nodes_extra_list(tree.right, res)


@remove
def preorder_get_all_nodes(tree: BinaryTree) -> List[int]:
    res = [tree.val]
    if tree.has_left():
        res += preorder_get_all_nodes(tree.left)
    if tree.has_right():
        res += preorder_get_all_nodes(tree.right)
    return res


@empty
# Returns a list of the values with an inorder traversal
def inorder_traversal(tree: BinaryTree) -> List[int]:
    # Returns a list with the results
    return inorder_get_all_nodes(tree)
    # Takes a list as parameter
    # res = []
    # inorder_get_all_nodes_extra_list(tree, res)
    # return res


@remove
def inorder_get_all_nodes_extra_list(tree: BinaryTree, res: List[int]):
    if tree.has_left():
        inorder_get_all_nodes_extra_list(tree.left, res)
    res.append(tree.val)
    if tree.has_right():
        inorder_get_all_nodes_extra_list(tree.right, res)


@remove
def inorder_get_all_nodes(tree: BinaryTree) -> List[int]:
    res = []
    if tree.has_left():
        res += inorder_get_all_nodes(tree.left)
    res.append(tree.val)
    if tree.has_right():
        res += inorder_get_all_nodes(tree.right)
    return res


@empty
# Returns a list of the values with a postorder traversal
def postorder_traversal(tree: BinaryTree) -> List[int]:
    # Returns a list with the results
    return postorder_get_all_nodes(tree)
    # Takes a list as parameter
    # res = []
    # postorder_get_all_nodes_extra_list(tree, res)
    # return res


@remove
def postorder_get_all_nodes_extra_list(tree: BinaryTree, res: List[int]):
    if tree.has_left():
        postorder_get_all_nodes_extra_list(tree.left, res)
    if tree.has_right():
        postorder_get_all_nodes_extra_list(tree.right, res)
    res.append(tree.val)


@remove
def postorder_get_all_nodes(tree: BinaryTree) -> List[int]:
    res = []
    if tree.has_left():
        res += postorder_get_all_nodes(tree.left)
    if tree.has_right():
        res += postorder_get_all_nodes(tree.right)
    res.append(tree.val)
    return res
