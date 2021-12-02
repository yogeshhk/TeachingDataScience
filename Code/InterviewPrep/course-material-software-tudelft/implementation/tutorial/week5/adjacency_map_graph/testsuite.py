import unittest
from weblabTestRunner import TestRunner
from decorators import yourtest, spectest

from .solution import Graph, Vertex


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        g = Graph()
        g.add_vertex()
        g.add_vertex()
        g.add_vertex()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        self.assertEqual(True, g.contains_edge(1, 0))
        self.assertEqual(False, g.contains_edge(2, 1))
        self.assertEqual(True, g.contains_edge(0, 2))
        g.remove_edge(2, 0)
        self.assertEqual(False, g.contains_edge(0, 2))
        self.assertEqual(False, g.contains_edge(2, 0))

    @yourtest
    @spectest(1)
    def test_add_vertex(self):
        g = Graph()
        self.assertEqual(0, g.size)
        g.add_vertex()
        self.assertEqual(0, g.vertices[0].idx)
        self.assertEqual(1, g.size)

    @yourtest
    @spectest(1)
    def test_add_edge(self):
        g = Graph()
        v0 = Vertex(0)
        v1 = Vertex(1)
        g.vertices[0] = v0
        g.vertices[1] = v1
        g.size = 2
        g.add_edge(0, 1)
        self.assertEqual({1: v1}, g.vertices[0].adj)
        self.assertEqual(1, len(g.vertices[0].adj))
        self.assertEqual({0: v0}, g.vertices[1].adj)
        self.assertEqual(1, len(g.vertices[1].adj))

    @yourtest
    @spectest(1)
    def test_remove_edge(self):
        g = Graph()
        v0 = Vertex(0)
        v1 = Vertex(1)
        g.vertices[0] = v0
        g.vertices[1] = v1
        g.size = 2
        v0.adj[v1.idx] = v1
        v1.adj[v0.idx] = v0
        self.assertEqual({1: v1}, g.vertices[0].adj)
        self.assertEqual(1, len(g.vertices[0].adj))
        self.assertEqual({0: v0}, g.vertices[1].adj)
        self.assertEqual(1, len(g.vertices[1].adj))
        g.remove_edge(0, 1)
        self.assertEqual({}, g.vertices[0].adj)
        self.assertEqual(0, len(g.vertices[0].adj))
        self.assertEqual({}, g.vertices[1].adj)
        self.assertEqual(0, len(g.vertices[1].adj))

    @yourtest
    @spectest(1)
    def test_contains_edge(self):
        g = Graph()
        v0 = Vertex(0)
        v1 = Vertex(1)
        g.vertices[0] = v0
        g.vertices[1] = v1
        g.size = 2
        self.assertFalse(g.contains_edge(1, 0))
        self.assertFalse(g.contains_edge(0, 1))
        v0.adj[v1.idx] = v1
        v1.adj[v0.idx] = v0
        self.assertTrue(g.contains_edge(0, 1))
        self.assertTrue(g.contains_edge(1, 0))

    @spectest(1)
    def test_path(self):
        n = 10
        g = Graph()
        for _ in range(n):
            g.add_vertex()
        for i in range(n - 1):
            g.add_edge(i, i + 1)
        for i in range(n - 1):
            self.assertEqual(False, g.contains_edge(i, i))
            self.assertEqual(False, g.contains_edge(i + 1, i + 1))
            self.assertEqual(True, g.contains_edge(i, i + 1))
            self.assertEqual(True, g.contains_edge(i + 1, i))

    @spectest(1)
    def test_complete_graph(self):
        n = 20
        g = Graph()
        for _ in range(n):
            g.add_vertex()
        for i in range(n):
            for j in range(n):
                g.add_edge(i, j)
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.assertEqual(False, g.contains_edge(i, j))
                else:
                    self.assertEqual(True, g.contains_edge(i, j))

    @spectest(1)
    def test_large(self):
        g = Graph()
        g.add_vertex()
        g.add_vertex()
        g.add_edge(0, 1)
        self.assertEqual(True, g.contains_edge(1, 0))
        g.add_vertex()
        self.assertEqual(False, g.contains_edge(2, 1))
        self.assertEqual(False, g.contains_edge(0, 2))
        g.add_vertex()
        g.add_vertex()
        g.add_vertex()
        g.add_edge(2, 5)
        g.add_edge(1, 4)
        g.add_edge(2, 1)
        g.add_edge(3, 0)
        g.add_edge(0, 5)
        self.assertEqual(True, g.contains_edge(5, 0))
        self.assertEqual(False, g.contains_edge(3, 1))
        self.assertEqual(True, g.contains_edge(3, 0))
        self.assertEqual(True, g.contains_edge(0, 3))
        self.assertEqual(True, g.contains_edge(5, 2))
        self.assertEqual(False, g.contains_edge(5, 1))
        g.remove_edge(1, 0)
        self.assertEqual(False, g.contains_edge(1, 0))
        self.assertEqual(False, g.contains_edge(0, 1))
        g.remove_edge(5, 0)
        self.assertEqual(False, g.contains_edge(5, 0))
        self.assertEqual(False, g.contains_edge(0, 5))


if __name__ == '__main__':
    unittest.main(testRunner=TestRunner)
