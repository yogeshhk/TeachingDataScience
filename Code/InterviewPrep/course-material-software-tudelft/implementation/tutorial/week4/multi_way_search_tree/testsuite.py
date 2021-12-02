import unittest

from decorators import spectest, yourtest, timeout, solution_only
from .solution import search
from .library import MWSTree, MWSTreeNode
from weblabTestRunner import TestRunner


if solution_only:
    huge_tree = MWSTree()
    k = list(range(1, 100000, 100))
    huge_tree.root.keys = k
    huge_tree.root.children.append(MWSTreeNode([], []))
    for i in k:
        ck = list(range(i + 1, i + 100))
        c = MWSTreeNode(ck, [])
        huge_tree.root.children.append(c)


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        tree = MWSTree()
        tree.root.keys = [10, 20, 30]
        c1 = MWSTreeNode([1, 2], [])
        c2 = MWSTreeNode([12, 15], [])
        c3 = MWSTreeNode([21, 26], [])
        c4 = MWSTreeNode([34], [])
        tree.root.children = [c1, c2, c3, c4]
        self.assertEqual(search(tree, 10), True)
        self.assertEqual(search(tree, 1), True)
        self.assertEqual(search(tree, 3), False)
        self.assertEqual(search(tree, 26), True)
        self.assertEqual(search(tree, 1125), False)

    @yourtest
    @spectest(1)
    def test_empty(self):
        tree = MWSTree()
        self.assertEqual(search(tree, 42), False)
        self.assertEqual(search(tree, 144), False)

    @yourtest
    @spectest(1)
    def test_one_element(self):
        tree = MWSTree()
        tree.root.keys = [42]
        self.assertEqual(search(tree, 42), True)
        self.assertEqual(search(tree, 41), False)

    @spectest(1)
    def test_large(self):
        tree = MWSTree()
        tree.root.keys = [52, 169, 1532]

        c1a = MWSTreeNode([1], [])
        c1b = MWSTreeNode([8, 13], [])
        c1c = MWSTreeNode([42], [])
        c1 = MWSTreeNode([6, 21], [c1a, c1b, c1c])

        c2a = MWSTreeNode([55], [])
        c2b = MWSTreeNode([74], [])
        c2c = MWSTreeNode([89], [])
        c2d = MWSTreeNode([98], [])
        c2 = MWSTreeNode([60, 88, 90], [c2a, c2b, c2c, c2d])

        c3a = MWSTreeNode([170], [])
        c3b = MWSTreeNode([180], [])
        c3c = MWSTreeNode([1337], [])
        c3 = MWSTreeNode([170, 1010], [c3a, c3b, c3c])

        c4a = MWSTreeNode([1897], [])
        c4b = MWSTreeNode([10000], [])
        c4 = MWSTreeNode([2440], [c4a, c4b])
        tree.root.children = [c1, c2, c3, c4]

        self.assertEqual(search(tree, 42), True)
        self.assertEqual(search(tree, 1010), True)
        self.assertEqual(search(tree, 10000), True)
        self.assertEqual(search(tree, 2440), True)
        self.assertEqual(search(tree, 1532), True)
        self.assertEqual(search(tree, 55), True)
        self.assertEqual(search(tree, 74), True)
        self.assertEqual(search(tree, 1337), True)

        self.assertEqual(search(tree, 144), False)
        self.assertEqual(search(tree, 24124), False)
        self.assertEqual(search(tree, 3), False)

    @spectest(1)
    @timeout(0.5)
    def test_efficiency(self):
        self.assertEqual(search(huge_tree, 100000), True)
        self.assertEqual(search(huge_tree, 99902), True)
        self.assertEqual(search(huge_tree, 42), True)
        self.assertEqual(search(huge_tree, 100001), False)
        self.assertEqual(search(huge_tree, 0), False)


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
