import random
import unittest
from typing import List

from .solution import topological_ordering
from .library import Graph
from weblabTestRunner import TestRunner
from decorators import yourtest, spectest


def check_topological_order(xs: List[int], g: Graph) -> bool:
    edges = g.all_edges()
    for (a, b) in edges:
        if xs.index(a) > xs.index(b):
            return False
    return True


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        g = Graph()
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        self.assertEqual(True, check_topological_order(topological_ordering(g), g))

    @yourtest
    @spectest(1)
    def test_one_node(self):
        g = Graph()
        g.add_node()
        self.assertEqual([0], topological_ordering(g))
        self.assertEqual(True, check_topological_order(topological_ordering(g), g))

    @yourtest
    @spectest(1)
    def test_empty(self):
        g = Graph()
        self.assertEqual([], topological_ordering(g))
        self.assertEqual(True, check_topological_order(topological_ordering(g), g))

    @yourtest
    @spectest(1)
    def test_small(self):
        g = Graph()
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_edge(2, 1)
        g.add_edge(2, 4)
        g.add_edge(4, 0)
        g.add_edge(0, 3)
        g.add_edge(1, 0)
        self.assertEqual(True, check_topological_order(topological_ordering(g), g))

    @spectest(1)
    def test_long_path(self):
        g = Graph()
        n = 42
        for i in range(n):
            g.add_node()
        g.add_node()
        g.add_edge(n, 0)
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        self.assertEqual(True, check_topological_order(topological_ordering(g), g))

    @spectest(1)
    def test_large(self):
        g = Graph()
        random.seed(42424242)
        n = 20
        for i in range(n):
            g.add_node()
        for i in range(n - 1):
            for j in range(8):
                g.add_edge(i, random.randrange(i + 1, n, 1))
        self.assertEqual(True, check_topological_order(topological_ordering(g), g))


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
