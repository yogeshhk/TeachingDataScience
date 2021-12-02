import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import is_distinct


class TestSolution(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_true(self):
        self.assertTrue(is_distinct([1, 2, 3, 4, 5]))

    @yourtest
    @spectest(1)
    def test_false(self):
        self.assertFalse(is_distinct([1, 2, 3, 1]))

    @spectest(1)
    def test_larger_true(self):
        self.assertTrue(is_distinct([9, 10, 1, 8, 5, 0, 2]))

    @spectest(1)
    def test_larger_false(self):
        self.assertFalse(is_distinct([9, 10, 1, 8, 5, 0, 2, 10]))

    @spectest(1)
    def test_large(self):
        self.assertFalse(is_distinct([5, 4, 2, 6, 0, 2, 0, 6, 2, 3, 3, 5, 6, 4, 3]))

    @spectest(1)
    def test_one_element(self):
        self.assertTrue(is_distinct([1]))

    @spectest(1)
    def test_massive(self):
        self.assertTrue(is_distinct(list(range(1000))))

    @spectest(1)
    def test_small_false(self):
        self.assertFalse(is_distinct([-1, -1]))

    @spectest(1)
    def test_small_true(self):
        self.assertTrue(is_distinct([-1, 1]))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
