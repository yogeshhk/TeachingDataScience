import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import generate_list


class TestSolution(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_equal(self):
        self.assertEqual([0, 2, 6, 12, 20, 30, 42, 56, 72, 90], generate_list())


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
