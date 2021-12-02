import unittest
import random

from weblabTestRunner import TestRunner
from decorators import yourtest, spectest

from .solution import Graph


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example_add_edge(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        self.assertEqual([[False, True, False, False],
                          [True, False, True, True],
                          [False, True, False, True],
                          [False, True, True, False]], g.adj)

    @yourtest
    @spectest(1)
    def test_example_add_vertex(self):
        g = Graph(1)
        self.assertEqual([[False]], g.adj)
        g.add_vertex()
        self.assertEqual([[False, False],
                          [False, False]], g.adj)

    @yourtest
    @spectest(1)
    def test_example_remove_edge(self):
        g = Graph(3)
        self.assertEqual([[False, False, False],
                          [False, False, False],
                          [False, False, False]], g.adj)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        self.assertEqual([[False, True, False],
                          [True, False, True],
                          [False, True, False]], g.adj)
        g.remove_edge(0, 1)
        self.assertEqual([[False, False, False],
                          [False, False, True],
                          [False, True, False]], g.adj)
        g.remove_edge(2, 1)
        self.assertEqual([[False, False, False],
                          [False, False, False],
                          [False, False, False]], g.adj)

    @yourtest
    @spectest(1)
    def test_example_contain_edge(self):
        g = Graph(4)
        g.add_edge(1, 0)
        g.add_edge(3, 0)
        g.add_edge(2, 1)
        self.assertTrue(g.contains_edge(1, 0))
        self.assertTrue(g.contains_edge(0, 1))
        self.assertFalse(g.contains_edge(0, 0))
        self.assertTrue(g.contains_edge(3, 0))
        self.assertTrue(g.contains_edge(0, 3))
        self.assertTrue(g.contains_edge(1, 2))
        self.assertTrue(g.contains_edge(2, 1))
        self.assertFalse(g.contains_edge(3, 2))
        self.assertFalse(g.contains_edge(2, 3))

    @yourtest
    @spectest(1)
    def test_self_edge(self):
        g = Graph(1)
        self.assertEqual([[False]], g.adj)
        self.assertFalse(g.contains_edge(0, 0))
        g.add_edge(0, 0)
        self.assertEqual([[True]], g.adj)
        self.assertTrue(g.contains_edge(0, 0))

    @spectest(1)
    def test_complete_graph(self):
        n = 20
        g = Graph(n)
        for i in range(n):
            for j in range(n):
                g.add_edge(i, j)
        res = [[True for _ in range(n)] for _ in range(n)]
        self.assertEqual(res, g.adj)
        for i in range(n):
            for j in range(n):
                self.assertTrue(g.contains_edge(i, j))

    @spectest(1)
    def test_path(self):
        n = 10
        g = Graph(n)
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        res = [[False for _ in range(n)] for _ in range(n)]
        for i in range(n - 1):
            res[i][i + 1] = True
            res[i + 1][i] = True
        self.assertEqual(res, g.adj)

    @spectest(1)
    def test_large(self):
        n = 10
        g = Graph(0)
        for _ in range(n):
            g.add_vertex()
        random.seed(424242)
        edges = []
        for _ in range(2*n):
            a = random.randrange(n)
            b = random.randrange(n)
            g.add_edge(a, b)
            edges.append((a, b))
        res = [[False, False, True, False, True, False, False, True, False, False],
               [False, False, False, False, False, False, True, False, False, False],
               [True, False, False, False, False, False, True, True, True, False],
               [False, False, False, False, False, True, True, True, True, True],
               [True, False, False, False, False, False, True, False, False, False],
               [False, False, False, True, False, False, True, False, True, False],
               [False, True, True, True, True, True, False, False, True, True],
               [True, False, True, True, False, False, False, True, False, False],
               [False, False, True, True, False, True, True, False, True, False],
               [False, False, False, True, False, False, True, False, False, False]]
        self.assertEqual(res, g.adj)
        for (a, b) in edges:
            self.assertTrue(g.contains_edge(a, b))
            self.assertTrue(g.contains_edge(b, a))


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
