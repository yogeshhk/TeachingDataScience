import unittest
from collections import deque
from typing import List

from .library import Graph, Node
from weblabTestRunner import TestRunner
from decorators import yourtest, spectest, remove
from .solution import connections


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        """
        Tests the following graph, starting from 0:
        1 - 0 - 3
            |
            2 - 4
        """
        g = Graph()
        for x in range(5):
            g.add_node(x)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(2, 4)
        g.add_edge(0, 3)
        self.assertEqual({1, 2, 3, 4}, connections(g, 4, g.get_node(0)))
        self.assertEqual({1, 2, 3}, connections(g, 3, g.get_node(0)))
        self.assertEqual({1, 2}, connections(g, 2, g.get_node(0)))
        self.assertEqual({1}, connections(g, 1, g.get_node(0)))

    @spectest(1)
    def test_one_node(self):
        g = Graph()
        g.add_node(1)
        self.assertEqual(set(), connections(g, 5, g.get_node(1)))

    @spectest(1)
    def test_two_nodes(self):
        g = Graph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)
        self.assertEqual({1}, connections(g, 1, g.get_node(0)))
        self.assertEqual({0}, connections(g, 1, g.get_node(1)))

    @spectest(1)
    def test_four_nodes(self):
        """
        0 - 1
        |   |
        2 - 3
        """
        g = Graph()
        for x in range(4):
            g.add_node(x)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        self.assertEqual({1}, connections(g, 1, g.get_node(0)))
        self.assertEqual({1, 2}, connections(g, 2, g.get_node(0)))
        self.assertEqual({1, 2, 3}, connections(g, 3, g.get_node(0)))
        self.assertEqual({0}, connections(g, 1, g.get_node(2)))
        self.assertEqual({0, 3}, connections(g, 2, g.get_node(2)))
        self.assertEqual({0, 1, 3}, connections(g, 3, g.get_node(2)))

    @spectest(5)
    def test_full_graph(self):
        r"""
        0 - 1 - 2
          /   \
        4 - 8 - 5 - 6
        | \ |
        3   7
        """
        g = Graph()
        for x in range(9):
            g.add_node(x)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(7, 8)
        g.add_edge(4, 3)
        g.add_edge(4, 7)
        g.add_edge(5, 6)
        g.add_edge(1, 4)
        g.add_edge(4, 8)
        g.add_edge(8, 5)
        g.add_edge(1, 5)
        for x in range(9):
            bfs_list = self.breadth_first(g.get_node(x))
            bfs_list = bfs_list[1:]
            for i in range(1, 9):
                if i != x:
                    self.assertEqual(set(bfs_list[:i]), connections(g, i, g.get_node(x)))

    @staticmethod
    @remove
    def breadth_first(n: Node) -> List[Node]:
        q = deque()
        q.append(n)
        seen = {n}
        res = []
        while not len(q) == 0:
            cur = q.popleft()
            res.append(cur.val)
            for node in cur.get_neighbours():
                if node not in seen:
                    q.append(node)
                    seen.add(node)
        return res


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
