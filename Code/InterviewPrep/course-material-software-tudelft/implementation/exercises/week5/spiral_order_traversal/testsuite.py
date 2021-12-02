import unittest
from collections import deque

from decorators import spectest, yourtest, remove, timeout
from weblabTestRunner import TestRunner
from .library import BinaryTree, Queue
from .solution import traversal


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        tree = BinaryTree(1,
                          BinaryTree(2,
                                     BinaryTree(4),
                                     BinaryTree(5)),
                          BinaryTree(3,
                                     BinaryTree(6),
                                     BinaryTree(7)))
        self.assertEqual([1, 3, 2, 4, 5, 6, 7], traversal(tree))

    @yourtest
    @spectest(1)
    def test_example_large(self):
        tree = BinaryTree(20,
                          BinaryTree(15,
                                     BinaryTree(12,
                                                BinaryTree(6),
                                                BinaryTree(14)),
                                     BinaryTree(10,
                                                BinaryTree(17),
                                                BinaryTree(19))),
                          BinaryTree(25,
                                     BinaryTree(22,
                                                BinaryTree(21),
                                                BinaryTree(24)),
                                     BinaryTree(20,
                                                BinaryTree(26),
                                                BinaryTree(32))))
        self.assertEqual([20, 25, 15, 12, 10, 22, 20, 32, 26, 24, 21, 19, 17, 14, 6], traversal(tree))

    @spectest(1)
    def test_small(self):
        tree = BinaryTree(1,
                          BinaryTree(2,
                                     BinaryTree(4,
                                                BinaryTree(8),
                                                BinaryTree(9)),
                                     BinaryTree(5,
                                                BinaryTree(10),
                                                BinaryTree(11))),
                          BinaryTree(3,
                                     BinaryTree(6,
                                                BinaryTree(12),
                                                BinaryTree(13)),
                                     BinaryTree(7,
                                                BinaryTree(14),
                                                BinaryTree(15))))
        self.assertEqual([1, 3, 2, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8], traversal(tree))

    @spectest(1)
    def test_large(self):
        tree = BinaryTree(42,
                          BinaryTree(21,
                                     BinaryTree(10,
                                                BinaryTree(5),
                                                None),
                                     BinaryTree(31)),
                          BinaryTree(63,
                                     BinaryTree(52,
                                                BinaryTree(47),
                                                None),
                                     BinaryTree(84)))
        self.assertEqual([42, 63, 21, 10, 31, 52, 84, 47, 5], traversal(tree))

    @spectest(1)
    def test_skew_left(self):
        tree = BinaryTree(42,
                          BinaryTree(36,
                                     BinaryTree(42,
                                                BinaryTree(121,
                                                           BinaryTree(1337),
                                                           None),
                                                None),
                                     None),
                          None)
        self.assertEqual([42, 36, 42, 121, 1337], traversal(tree))

    @spectest(1)
    def test_skew_right(self):
        tree = BinaryTree(42,
                          None,
                          BinaryTree(36,
                                     None,
                                     BinaryTree(421,
                                                None,
                                                BinaryTree(121,
                                                           None,
                                                           BinaryTree(1337)))))
        self.assertEqual([42, 36, 421, 121, 1337], traversal(tree))

    @spectest(1)
    @timeout(1)
    def test_complete_tree(self):
        tree = self.build_complete_tree(50)
        self.assertEqual([1, 3, 2, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32],
                         traversal(tree))

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
