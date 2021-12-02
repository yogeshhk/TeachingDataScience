import unittest
from collections import deque

from decorators import spectest, yourtest, remove, timeout
from weblabTestRunner import TestRunner
from .library import BinaryTree
from .solution import preorder_traversal, inorder_traversal, \
    postorder_traversal


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_small_preorder(self):
        tree = BinaryTree(1, BinaryTree(2), BinaryTree(3))
        self.assertEqual([1, 2, 3], preorder_traversal(tree))

    @yourtest
    @spectest(1)
    def test_small_inorder(self):
        tree = BinaryTree(1, BinaryTree(2), BinaryTree(3))
        self.assertEqual([2, 1, 3], inorder_traversal(tree))

    @yourtest
    @spectest(1)
    def test_small_postorder(self):
        tree = BinaryTree(1, BinaryTree(2), BinaryTree(3))
        self.assertEqual([2, 3, 1], postorder_traversal(tree))

    @spectest(1)
    def test_large_preorder(self):
        tree = BinaryTree(42, BinaryTree(21, BinaryTree(10, BinaryTree(5), None), BinaryTree(31)),
                          BinaryTree(63, BinaryTree(52, BinaryTree(47), None), BinaryTree(84)))
        self.assertEqual([42, 21, 10, 5, 31, 63, 52, 47, 84], preorder_traversal(tree))

    @spectest(1)
    def test_large_inorder(self):
        tree = BinaryTree(42, BinaryTree(21, BinaryTree(10, BinaryTree(5), None), BinaryTree(31)),
                          BinaryTree(63, BinaryTree(52, BinaryTree(47), None), BinaryTree(84)))
        self.assertEqual([5, 10, 21, 31, 42, 47, 52, 63, 84], inorder_traversal(tree))

    @spectest(1)
    def test_large_postorder(self):
        tree = BinaryTree(42, BinaryTree(21, BinaryTree(10, BinaryTree(5), None), BinaryTree(31)),
                          BinaryTree(63, BinaryTree(52, BinaryTree(47), None), BinaryTree(84)))
        self.assertEqual([5, 10, 31, 21, 47, 52, 84, 63, 42], postorder_traversal(tree))

    @spectest(1)
    def test_skew_left_preorder(self):
        tree = BinaryTree(42,
                          BinaryTree(36,
                                     BinaryTree(42,
                                                BinaryTree(121,
                                                           BinaryTree(1337),
                                                           None),
                                                None),
                                     None),
                          None)
        self.assertEqual([42, 36, 42, 121, 1337], preorder_traversal(tree))

    @spectest(1)
    def test_skew_left_inorder(self):
        tree = BinaryTree(42,
                          BinaryTree(36,
                                     BinaryTree(42,
                                                BinaryTree(121,
                                                           BinaryTree(1337),
                                                           None),
                                                None),
                                     None),
                          None)
        self.assertEqual([1337, 121, 42, 36, 42], inorder_traversal(tree))

    @spectest(1)
    def test_skew_left_postorder(self):
        tree = BinaryTree(42,
                          BinaryTree(36,
                                     BinaryTree(42,
                                                BinaryTree(121,
                                                           BinaryTree(1337),
                                                           None),
                                                None),
                                     None),
                          None)
        self.assertEqual([1337, 121, 42, 36, 42], postorder_traversal(tree))

    @spectest(1)
    def test_skew_right_preorder(self):
        tree = BinaryTree(42,
                          None,
                          BinaryTree(36,
                                     None,
                                     BinaryTree(421,
                                                None,
                                                BinaryTree(121,
                                                           None,
                                                           BinaryTree(1337)))))
        self.assertEqual([42, 36, 421, 121, 1337], preorder_traversal(tree))

    @spectest(1)
    def test_skew_right_inorder(self):
        tree = BinaryTree(42,
                          None,
                          BinaryTree(36,
                                     None,
                                     BinaryTree(421,
                                                None,
                                                BinaryTree(121,
                                                           None,
                                                           BinaryTree(1337)))))
        self.assertEqual([42, 36, 421, 121, 1337], inorder_traversal(tree))

    @spectest(1)
    def test_skew_right_postorder(self):
        tree = BinaryTree(42,
                          None,
                          BinaryTree(36,
                                     None,
                                     BinaryTree(421,
                                                None,
                                                BinaryTree(121,
                                                           None,
                                                           BinaryTree(1337)))))
        self.assertEqual([1337, 121, 421, 36, 42], postorder_traversal(tree))

    @spectest(1)
    @timeout(1)
    def test_complete_tree_preorder(self):
        tree = self.build_complete_tree(50)
        self.assertEqual([1, 2, 4, 8, 16, 32, 33, 17, 34, 35, 9, 18, 36, 37, 19, 38, 39, 5, 10, 20, 40, 41, 21, 42, 43,
                          11, 22, 44, 45, 23, 46, 47, 3, 6, 12, 24, 48, 49, 25, 13, 26, 27, 7, 14, 28, 29, 15, 30, 31],
                         preorder_traversal(tree))

    @spectest(1)
    @timeout(1)
    def test_complete_tree_inorder(self):
        tree = self.build_complete_tree(50)
        self.assertEqual([32, 16, 33, 8, 34, 17, 35, 4, 36, 18, 37, 9, 38, 19, 39, 2, 40, 20, 41, 10, 42, 21, 43, 5, 44,
                          22, 45, 11, 46, 23, 47, 1, 48, 24, 49, 12, 25, 6, 26, 13, 27, 3, 28, 14, 29, 7, 30, 15, 31],
                         inorder_traversal(tree))

    @spectest(1)
    @timeout(1)
    def test_complete_tree_postorder(self):
        tree = self.build_complete_tree(50)
        self.assertEqual([32, 33, 16, 34, 35, 17, 8, 36, 37, 18, 38, 39, 19, 9, 4, 40, 41, 20, 42, 43, 21, 10, 44, 45,
                          22, 46, 47, 23, 11, 5, 2, 48, 49, 24, 25, 12, 26, 27, 13, 6, 28, 29, 14, 30, 31, 15, 7, 3, 1],
                         postorder_traversal(tree))

    @classmethod
    @remove
    # Creates a complete tree with n nodes
    def build_complete_tree(cls, n: int) -> BinaryTree:
        if n == 0:
            return None
        root = BinaryTree(1)
        q = deque()
        q.append(root)
        i = 2
        while i < n:
            p = q.popleft()
            p.left = BinaryTree(i)
            q.append(p.left)
            i += 1
            if i <= n:
                p.right = BinaryTree(i)
                q.append(p.right)
                i += 1

        return root


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
