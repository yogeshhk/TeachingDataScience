import unittest
from .library import Graph
from weblabTestRunner import TestRunner
from decorators import yourtest, spectest
from .solution import breadth_first


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        r"""
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
        self.assertEqual([0, 1, 2, 3, 4], breadth_first(g, g.get_node(0)))

    @spectest(1)
    def test_one_node(self):
        g = Graph()
        g.add_node(1)
        self.assertEqual([1], breadth_first(g, g.get_node(1)))

    @spectest(1)
    def test_two_nodes(self):
        g = Graph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)
        self.assertEqual([0, 1], breadth_first(g, g.get_node(0)))
        self.assertEqual([1, 0], breadth_first(g, g.get_node(1)))

    @spectest(1)
    def test_four_nodes(self):
        """
        0 - 1
        |   |
        2   3
        """
        g = Graph()
        for x in range(4):
            g.add_node(x)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        self.assertEqual([0, 1, 2, 3], breadth_first(g, g.get_node(0)))
        self.assertEqual([1, 0, 3, 2], breadth_first(g, g.get_node(1)))
        self.assertEqual([2, 0, 1, 3], breadth_first(g, g.get_node(2)))

    @spectest(1)
    def test_single_loop(self):
        g = Graph()
        g.add_node(16)
        g.add_edge(16, 16)
        self.assertEqual([16], breadth_first(g, g.get_node(16)))

    @spectest(5)
    def test_full_graph(self):
        r"""
        0  -  1 - 2
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
        self.assertEqual([0, 1, 2, 4, 5, 3, 7, 8, 6], breadth_first(g, g.get_node(0)))
        self.assertEqual([2, 1, 0, 4, 5, 3, 7, 8, 6], breadth_first(g, g.get_node(2)))
        self.assertEqual([7, 4, 8, 1, 3, 5, 0, 2, 6], breadth_first(g, g.get_node(7)))


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
