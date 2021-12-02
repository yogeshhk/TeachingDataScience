import unittest
from decorators import spectest, yourtest, solution_only, timeout
from weblabTestRunner import TestRunner
from .solution import Tree, Node
import random

if solution_only:
    massive_tree = Tree()
    random.seed(713563115)
    massive_tree_values = random.sample(range(100000), 70000)
    for val in massive_tree_values:
        massive_tree.add(val)

if solution_only:
    def test_invariant(self, node):
        # Tests that all nodes have at most 3 values and at most 4 children
        self.assertGreaterEqual(3, len(node.values))
        self.assertEqual(sorted(node.values), node.values)
        self.assertTrue(len(node.values) + 1 == len(node.children) or not node.children)
        for child in node.children:
            test_invariant(self, child)

if solution_only:
    def height(node) -> int:
        # Returns the height of the tree from the given node
        if not node.children:
            return 1
        return 1 + max([height(c) for c in node.children])


class TestSuite(unittest.TestCase):
    @yourtest
    @spectest(1)
    def test_example_add(self):
        tree = Tree()
        for x in range(3):
            tree.add(x)
        self.assertEqual(list(range(3)), tree.root.values)

    @yourtest
    @spectest(1)
    def test_split_example(self):
        r"""
        Tests the following tree:
             [20,        100]
            /         |       \
        [3, 10, 17] [30]    [179]

        We add 5:
             [10,  20,   100]
            /    |    |      \
        [3, 5] [17] [30]    [179]
        """
        root = Node(None, [20, 100])
        node1 = Node(root, [3, 10, 17])
        node2 = Node(root, [30])
        node3 = Node(root, [179])
        root.children = [node1, node2, node3]
        tree = Tree()
        tree.root = root
        tree.add(5)
        self.assertEqual([10, 20, 100], tree.root.values)
        self.assertEqual([3, 5], tree.root.children[0].values)
        self.assertEqual([17], tree.root.children[1].values)
        self.assertEqual([30], tree.root.children[2].values)
        self.assertEqual([179], tree.root.children[3].values)

    @yourtest
    @spectest(1)
    def test_split_children(self):
        r"""
        Tests the following tree:
                 [20,              100]
                /               |       \
            [3, 10, 17]       [30]    [179]
           /  |   |   \      /  \     |    \
        [2] [5] [12] [19]  [23] [51] [116] [189]

        We add 7:
                 [10.   20,          100]
                /       |          |       \
            [3]       [17]       [30]    [179]
           /  |      |   \       /  \     |    \
        [2] [5, 7] [12] [19]  [23] [51] [116] [189]
        """
        root = Node(None, [20, 100])
        node1 = Node(root, [3, 10, 17])
        node2 = Node(root, [30])
        node3 = Node(root, [179])
        node4 = Node(node1, [2])
        node5 = Node(node1, [5])
        node6 = Node(node1, [12])
        node7 = Node(node1, [19])
        node8 = Node(node2, [23])
        node9 = Node(node2, [51])
        node10 = Node(node2, [116])
        node11 = Node(node2, [189])
        root.children = [node1, node2, node3]
        node1.children = [node4, node5, node6, node7]
        node2.children = [node8, node9]
        node3.children = [node10, node11]
        tree = Tree()
        tree.root = root
        tree.add(7)
        self.assertEqual([10, 20, 100], tree.root.values)
        self.assertEqual([3], tree.root.children[0].values)
        self.assertEqual([17], tree.root.children[1].values)
        self.assertEqual([30], tree.root.children[2].values)
        self.assertEqual([179], tree.root.children[3].values)
        self.assertEqual([2], tree.root.children[0].children[0].values)
        self.assertEqual([5, 7], tree.root.children[0].children[1].values)
        self.assertEqual([12], tree.root.children[1].children[0].values)
        self.assertEqual([19], tree.root.children[1].children[1].values)
        self.assertEqual([23], tree.root.children[2].children[0].values)
        self.assertEqual([51], tree.root.children[2].children[1].values)
        self.assertEqual([116], tree.root.children[3].children[0].values)
        self.assertEqual([189], tree.root.children[3].children[1].values)

    @spectest(2)
    def test_split(self):
        node1 = Node(None, [3, 5, 7])
        node2 = Node(node1, [2])
        node3 = Node(node1, [4])
        node4 = Node(node1, [6])
        node5 = Node(node1, [8])
        node1.children = [node2, node3, node4, node5]
        node1.split()
        root = node1.parent
        self.assertEqual([5], root.values)
        self.assertEqual([3], root.children[0].values)
        self.assertEqual([7], root.children[1].values)
        self.assertEqual([2], root.children[0].children[0].values)
        self.assertEqual([4], root.children[0].children[1].values)
        self.assertEqual([6], root.children[1].children[0].values)
        self.assertEqual([8], root.children[1].children[1].values)

    @yourtest
    @spectest(1)
    def test_root_split(self):
        tree = Tree()
        for x in range(4):
            tree.add(x)
        self.assertEqual([1], tree.root.values)
        self.assertEqual([0], tree.root.children[0].values)
        self.assertEqual([2, 3], tree.root.children[1].values)

    @spectest(2)
    def test_add(self):
        tree = Tree()
        for x in [8, 2, 4, 9, 10, 12]:
            tree.add(x)
        self.assertEqual([4, 9], tree.root.values)
        self.assertEqual([2], tree.root.children[0].values)
        self.assertEqual([8], tree.root.children[1].values)
        self.assertEqual([10, 12], tree.root.children[2].values)

    @yourtest
    @spectest(1)
    def test_example_contains(self):
        tree = Tree()
        values = [8, 2, 4, 10, 13, 17]
        for x in values:
            tree.add(x)
        for x in values:
            self.assertTrue(tree.contains(x))
            self.assertFalse(tree.contains(x + 1))
            self.assertFalse(tree.contains(x - 1))

    @spectest(1)
    def test_split_correct_parents(self):
        # Checks that the parents are correctly set after a split
        tree = Tree()
        for x in range(4):
            tree.add(x)
        self.assertEqual([1], tree.root.children[0].parent.values)
        self.assertEqual([1], tree.root.children[1].parent.values)

    @spectest(2)
    def test_add_contains_large(self):
        tree = Tree()
        random.seed(917410)
        values = random.sample(range(10000), 400)
        for x in values:
            tree.add(x)
        for x in values:
            self.assertTrue(tree.contains(x))

    @spectest(4)
    @timeout(4)
    def test_everything_large(self):
        tree = Tree()
        random.seed(4175913)
        values = random.sample(range(100000), 40000)
        for x in values:
            tree.add(x)
        for x in values:
            self.assertTrue(tree.contains(x))
        test_invariant(self, tree.root)
        self.assertEqual(12, height(tree.root))

    @spectest(3)
    @timeout(1.5)
    def test_add_complexity(self):
        tree = Tree()
        random.seed(8324612)
        values = random.sample(range(100000), 40000)
        for x in values:
            tree.add(x)
        self.assertEqual([44035], tree.root.values)
        test_invariant(self, tree.root)
        self.assertEqual(12, height(tree.root))

    @spectest(3)
    @timeout(2)
    def test_contains_complexity(self):
        self.assertEqual([46834], massive_tree.root.values)
        test_invariant(self, massive_tree.root)
        self.assertEqual(13, height(massive_tree.root))
        for x in massive_tree_values:
            self.assertTrue(massive_tree.contains(x))
        for x in [6, 9, 16, 20, 21, 27, 30, 34, 46, 47, 49, 65, 67, 72, 73, 74, 80, 88, 90, 91]:
            self.assertFalse(massive_tree.contains(x))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
