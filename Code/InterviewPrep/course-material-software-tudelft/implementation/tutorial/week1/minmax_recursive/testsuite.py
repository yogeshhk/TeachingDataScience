import random
import unittest

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import minmax


class TestSolution(unittest.TestCase):

    @spectest(1)
    @yourtest
    @timeout(1)
    def test_1_to_10(self):
        self.assertEqual((0, 10), minmax(list(range(11))))

    @spectest(1)
    @yourtest
    @timeout(1)
    def test_neg4_to_3(self):
        self.assertEqual((-4, 3), minmax(list(range(-4, 4))))

    @spectest(1)
    @yourtest
    @timeout(1)
    def test_unsorted(self):
        self.assertEqual((-2, 5), minmax([1, 3, -2, 4, 1, 5, -1, 3]))

    @spectest(5)
    @timeout(1)
    def test_random_positive(self):
        random.seed(164201)
        for x in range(100, 1000, 100):
            xs = random.sample(range(x * 2), x)
            self.assertEqual((min(xs), max(xs)), minmax(xs))

    @spectest(9)
    @timeout(1)
    def test_random_all(self):
        random.seed(98325)
        for x in range(100, 1000, 100):
            xs = random.sample(range(-x, x), x)
            self.assertEqual((min(xs), max(xs)), minmax(xs))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
