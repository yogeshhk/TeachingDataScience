import unittest

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import divide


class TestSolution(unittest.TestCase):

    @spectest(1)
    @yourtest
    @timeout(0.1)
    def test_simple(self):
        self.assertEqual((4, 2), divide(14, 3))

    @spectest(1)
    @timeout(0.1)
    def test_no_quotient(self):
        self.assertEqual((0, 1), divide(1, 2))

    @spectest(1)
    @timeout(0.1)
    def test_no_remainder(self):
        self.assertEqual((2, 0), divide(4, 2))

    @spectest(1)
    @timeout(0.1)
    def test_divide_zero(self):
        self.assertEqual((0, 0), divide(0, 6))

    @spectest(5)
    @timeout(2)
    def test_large_positive(self):
        for x in range(1500, 20000, 23):
            for y in range(1, 1000, 3):
                self.assertEqual((x // y, x % y), divide(x, y))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
