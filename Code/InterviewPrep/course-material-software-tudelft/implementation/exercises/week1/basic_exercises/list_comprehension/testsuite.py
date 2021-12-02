import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import iterative_function1, list_comprehension_function2, \
    list_comprehension_function1, iterative_function2


class TestSuite(unittest.TestCase):

    @yourtest
    def test_list_comprehension_function1(self):
        self.assertEqual(list_comprehension_function1(1, 3), [1, 4, 9])

    @yourtest
    def test_iterative_function2(self):
        self.assertEqual(iterative_function2(1, 3), [3, 6, 9])

    @spectest(1)
    def test_iterative_function1(self):
        self.assertEqual(iterative_function1(1, 5), [1, 4, 9, 16, 25])

    @spectest(1)
    def test_list_comprehension_function2(self):
        self.assertEqual(list_comprehension_function2(-3, 3), [-9, -6, -3, 0, 3, 6, 9])

    if __name__ == "__main__":
        unittest.main(testRunner=TestRunner)
