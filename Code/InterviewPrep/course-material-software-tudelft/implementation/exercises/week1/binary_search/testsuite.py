import unittest

from decorators import spectest, timeout, yourtest, solution_only
from weblabTestRunner import TestRunner
from .solution import binary_search

# LONG_LIST is created outside the test case to avoid being part of the searching time
if solution_only:
    LONG_LIST = list(range(1, 500_000))


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_find_empty_list(self):
        self.assertFalse(binary_search([], 42))

    @yourtest
    @spectest(1)
    def test_find_small_list_true(self):
        self.assertTrue(binary_search([1, 2, 4, 8, 16], 8))

    @yourtest
    @spectest(1)
    def test_find_small_list_false(self):
        self.assertFalse(binary_search([1, 2, 4, 8, 16], 9))

    @spectest(1)
    def test_find_too_small(self):
        self.assertTrue(binary_search([1], 1))
        self.assertFalse(binary_search([1, 2, 3], -3))

    @spectest(1)
    def test_find_too_big(self):
        self.assertTrue(binary_search([1], 1))
        self.assertFalse(binary_search([1, 2, 3], 42))

    @spectest(5)
    def test_big_even_odd(self):
        items_list = [2 * i for i in range(1, 100)]
        for i in range(-30, 0):
            self.assertFalse(binary_search(items_list, i))
        for i in range(1, 100):
            self.assertFalse(binary_search(items_list, 2 * i - 1))
            self.assertTrue(binary_search(items_list, 2 * i))
        for i in range(201, 250):
            self.assertFalse(binary_search(items_list, i))

    @spectest(10)
    @timeout(0.1)
    def test_log_time(self):
        self.assertTrue(binary_search([1], 1))
        # LONG_LIST is created outside the test case to avoid being part of the searching time
        self.assertFalse(binary_search(LONG_LIST, -42))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
