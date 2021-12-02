import random
import unittest
from decorators import spectest, timeout, yourtest
from weblabTestRunner import TestRunner
from .solution import selection_sort


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_small(self):
        xs = [2, 1, 3]
        selection_sort(xs)
        self.assertEqual(xs, [1, 2, 3])

    @spectest(1)
    def test_empty(self):
        xs = []
        selection_sort(xs)
        self.assertEqual(xs, [])

    @spectest(1)
    def test_one(self):
        xs = [42]
        selection_sort(xs)
        self.assertEqual(xs, [42])

    @spectest(1)
    def test_sorted(self):
        xs = list(i for i in range(24))
        selection_sort(xs)
        self.assertEqual(xs, list(i for i in range(24)))

    @spectest(1)
    def test_reverse(self):
        xs = list(i for i in range(23, 0, -1))
        selection_sort(xs)
        self.assertEqual(xs, list(i for i in range(1, 24)))

    @spectest(5)
    def test_small(self):
        xs = list(range(-40, 50, 3))
        sol = xs.copy()
        random.seed(49154)
        random.shuffle(xs)
        selection_sort(xs)
        self.assertEqual(xs, sol)

    @spectest(5)
    @timeout(2)
    def test_large(self):
        random.seed(691721)
        xs = [random.randint(-1000, 1000) for _ in range(5000)]
        sol = sorted(xs)
        selection_sort(xs)
        self.assertEqual(xs, sol)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
