import random
import unittest

from decorators import spectest, timeout, yourtest, solution_only
from weblabTestRunner import TestRunner

from .solution import sort

# huge and huge_sorted are created outside the test case to avoid being part of the searching time
if solution_only:
    random.seed(42)
    huge = list(random.randint(0, 10000) for _ in range(50000))
    huge_sorted = sorted(huge)


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        x = [2, 1, 3]
        self.assertEqual([1, 2, 3], sort(x))

    @spectest(1)
    def test_empty(self):
        x = []
        self.assertEqual([], sort(x))

    @spectest(1)
    def test_one(self):
        x = [42]
        self.assertEqual([42], sort(x))

    @spectest(1)
    def test_sorted(self):
        x = list(range(24))
        self.assertEqual(list(range(24)), x)

    @spectest(1)
    def test_reverse(self):
        x = list(reversed(range(1, 24)))
        self.assertEqual(list(range(1, 24)), sort(x))

    @spectest(5)
    def test_small(self):
        x = [2, 199, -23, 8, 31, -99, 2654, 274, 0]
        self.assertEqual([-99, -23, 0, 2, 8, 31, 199, 274, 2654], sort(x))

    @spectest(5)
    def test_large(self):
        random.seed(917481)
        x = [random.randint(0, 100) for _ in range(100)]
        y = sorted(x)
        self.assertEqual(y, sort(x))

    @spectest(10)
    @timeout(1)
    def test_huge(self):
        self.assertEqual(huge_sorted, sort(huge))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
