import unittest
from typing import List

from weblabTestRunner import TestRunner
from .library import BinaryTree
from .solution import search
from decorators import yourtest, spectest, timeout, solution_only
import random

if solution_only:
    # Creates a tree with the given nodes in the given order
    def build_tree(nodes: List[int]) -> BinaryTree:
        tree = BinaryTree(nodes[0])
        for x in nodes[1:]:
            add_to_tree(tree, x)
        return tree

if solution_only:
    # Adds the given value to the given tree
    def add_to_tree(tree: BinaryTree, val: int):
        if val <= tree.val:
            if tree.has_left():
                add_to_tree(tree.left, val)
            else:
                tree.left = BinaryTree(val)
        else:
            if tree.has_right():
                add_to_tree(tree.right, val)
            else:
                tree.right = BinaryTree(val)

if solution_only:
    random.seed(59175)
    massive_list = random.sample(range(-100000, 190000, 3), 2**16)
    massive_tree = build_tree(massive_list)


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example_1(self):
        tree = BinaryTree(10, BinaryTree(5), BinaryTree(15))
        self.assertTrue(search(5, tree))
        self.assertTrue(search(10, tree))
        self.assertTrue(search(15, tree))
        self.assertFalse(search(6, tree))
        self.assertFalse(search(12, tree))
        self.assertFalse(search(16, tree))

    @yourtest
    @spectest(1)
    def test_example_2(self):
        """
        ..........13.........
        .....5.........15....
        ..3....8.....14..18..
        """
        tree = BinaryTree(13, BinaryTree(5, BinaryTree(3), BinaryTree(8)),
                          BinaryTree(15, BinaryTree(14), BinaryTree(18)))
        for x in [3, 5, 8, 14, 15, 18]:
            self.assertTrue(search(x, tree))
        self.assertFalse(search(0, tree))
        self.assertFalse(search(10, tree))
        self.assertFalse(search(20, tree))

    @spectest(1)
    def test_only_root(self):
        self.assertTrue(search(42, BinaryTree(42)))
        self.assertFalse(search(1, BinaryTree(2)))

    @spectest(1)
    def test_only_left_child(self):
        self.assertTrue(search(2, (BinaryTree(1, None, BinaryTree(2)))))
        self.assertFalse(search(3, (BinaryTree(1, None, BinaryTree(2)))))

    @spectest(1)
    def test_only_right_child(self):
        self.assertTrue(search(2, BinaryTree(1, None, BinaryTree(2))))
        self.assertFalse(search(3, BinaryTree(1, None, BinaryTree(2))))

    @spectest(2)
    def test_two_levels(self):
        nodes = [41, 24, 21, 36, 47, 45, 61]
        tree = build_tree(nodes)
        for x in nodes:
            self.assertTrue(search(x, tree))
            self.assertFalse(search(x-1, tree))
            self.assertFalse(search(x + 1, tree))
            self.assertFalse(search(x * 2, tree))

    @spectest(2)
    def test_three_levels(self):
        """
        ...........42.........
        ......21........63....
        ....10..31....52..84..
        ...5........47........
        """
        nodes = [42, 21, 63, 10, 31, 52, 84, 5, 47]
        tree = build_tree(nodes)
        for x in nodes:
            self.assertTrue(search(x, tree))
            self.assertFalse(search(x - 1, tree))
            self.assertFalse(search(x + 1, tree))

    @spectest(3)
    def test_reverse(self):
        tree = build_tree(list(range(950, -1, -1)))
        self.assertTrue(search(0, tree))
        self.assertFalse(search(2000, tree))

    @spectest(3)
    def test_ordered(self):
        tree = build_tree(list(range(950)))
        self.assertTrue(search(0, tree))
        self.assertTrue(search(487, tree))
        self.assertFalse(search(950, tree))

    @spectest(6)
    @timeout(1.5)
    def test_massive(self):
        for x in massive_list:
            self.assertTrue(search(x, massive_tree))
        self.assertFalse(search(-100001, massive_tree))
        self.assertFalse(search(-9999, massive_tree))
        self.assertFalse(search(190001, massive_tree))


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
