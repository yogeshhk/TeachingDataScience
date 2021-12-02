import random
import unittest

from decorators import spectest, timeout, yourtest, solution_only
from weblabTestRunner import TestRunner

from .solution import sort

# huge and huge_sorted are created outside the test case to avoid being part of the searching time
if solution_only:
    random.seed(42)
    huge = list(random.randint(0, 10000) for r in range(3000))
    huge_sorted = huge.copy()
    huge_sorted.sort(reverse=True)


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        x = [2, 1, 3]
        sort(x)
        self.assertEqual(x, [3, 2, 1])

    @spectest(1)
    def test_empty(self):
        x = []
        sort(x)
        self.assertEqual(x, [])

    @spectest(1)
    def test_one(self):
        x = [42]
        sort(x)
        self.assertEqual(x, [42])

    @spectest(1)
    def test_sorted(self):
        x = list(i for i in range(24, -1, -1))
        sort(x)
        self.assertEqual(x, list(i for i in range(24, -1, -1)))

    @spectest(1)
    def test_reverse(self):
        x = list(i for i in range(24))
        sort(x)
        self.assertEqual(x, list(i for i in range(23, -1, -1)))

    @spectest(5)
    def test_small(self):
        x = [2, 199, -23, 8, 31, -99, 2654, 274, 0]
        sort(x)
        self.assertEqual(x, [2654, 274, 199, 31, 8, 2, 0, -23, -99])

    @spectest(5)
    def test_large(self):
        random.seed(42)
        x = [random.randint(0, 100) for _ in range(100)]
        y = x.copy()
        y.sort(reverse=True)
        sort(x)
        self.assertEqual(x, y)

    @spectest(10)
    @timeout(1)
    def test_huge(self):
        sort(huge)
        self.assertEqual(huge, huge_sorted)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
