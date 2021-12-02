import unittest
from collections import deque

from weblabTestRunner import TestRunner
from .library import BinaryTree
from .solution import count_leaves
from decorators import yourtest, spectest, remove, timeout


class TestSuite(unittest.TestCase):
    @yourtest
    @spectest(1)
    def test_example_1(self):
        tree = BinaryTree(1, BinaryTree(2), BinaryTree(3))
        self.assertEqual(2, count_leaves(tree))

    @yourtest
    @spectest(1)
    def test_example_2(self):
        tree = BinaryTree(1, BinaryTree(2), BinaryTree(3, BinaryTree(4, BinaryTree(6)), BinaryTree(6)))
        self.assertEqual(3, count_leaves(tree))

    @spectest(1)
    def test_only_root(self):
        self.assertEqual(1, count_leaves(BinaryTree(42)))

    @spectest(1)
    def test_only_left_child(self):
        self.assertEqual(1, count_leaves(BinaryTree(1, BinaryTree(2))))

    @spectest(1)
    def test_only_right_child(self):
        self.assertEqual(1, count_leaves(BinaryTree(1, None, BinaryTree(2))))

    @spectest(2)
    def test_two_levels(self):
        tree = BinaryTree(42, BinaryTree(36, BinaryTree(21), BinaryTree(24)),
                          BinaryTree(47, BinaryTree(97), BinaryTree(16)))
        self.assertEqual(4, count_leaves(tree))

    @spectest(2)
    def test_two_levels_skew_1(self):
        tree = BinaryTree(42, BinaryTree(36, BinaryTree(21), BinaryTree(24)), BinaryTree(47))
        self.assertEqual(3, count_leaves(tree))

    @spectest(2)
    def test_two_levels_skew_2(self):
        tree = BinaryTree(42, BinaryTree(36, BinaryTree(21), BinaryTree(24)), None)
        self.assertEqual(2, count_leaves(tree))

    @spectest(2)
    def test_three_levels(self):
        """
        ...........42.........
        ......21........63....
        ....10..31....52..84..
        ...5........47........
        """
        tree = BinaryTree(42, BinaryTree(21, BinaryTree(10, BinaryTree(5), None), BinaryTree(31)),
                          BinaryTree(63, BinaryTree(52, BinaryTree(47), None), BinaryTree(84)))
        self.assertEqual(4, count_leaves(tree))

    @spectest(3)
    def test_three_levels_skew(self):
        tree = BinaryTree(42, BinaryTree(21, BinaryTree(10, BinaryTree(5), None), BinaryTree(31)),
                          BinaryTree(63, BinaryTree(52, BinaryTree(47), None), BinaryTree(84)))
        self.assertEqual(4, count_leaves(tree))

    @spectest(5)
    def test_half_complete_tree(self):
        for i in range(2, 8):
            tree = self.build_complete_tree(2 ** i)
            self.assertEqual(2 ** (i - 1), count_leaves(tree))

    @spectest(10)
    @timeout(1.5)
    def test_big_complete_tree(self):
        for i in range(1, 1000):
            expected = i - i // 2
            tree = self.build_complete_tree(i)
            self.assertEqual(expected, count_leaves(tree))

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


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
