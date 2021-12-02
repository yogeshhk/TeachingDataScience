import random
import unittest
from typing import List

from decorators import remove, spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import even_before_odd


# Checks there is no even value after an odd value
def correct_order(xs: List[int]) -> bool:
    odd = False
    for x in range(len(xs)):
        if xs[x] % 2:
            odd = True
        elif odd:
            return False
    return True


@remove
# Runs a test case with a given list
def run_with_list(self, xs: List[int]):
    sol = xs.copy()
    even_before_odd(sol)
    self.assertTrue(correct_order(sol))
    self.assertCountEqual(xs, sol)


class TestSolution(unittest.TestCase):

    @spectest(1)
    @yourtest
    @timeout(1)
    def test_example(self):
        xs = [1, 2, 3, 4, 5]
        sol = [1, 2, 3, 4, 5]
        even_before_odd(sol)
        self.assertTrue(correct_order(sol))  # Checks that sol has all even numbers before odd numbers
        self.assertCountEqual(xs, sol)  # Checks that xs and sol still have the same elements, regardless of their order

    @spectest(1)
    @yourtest
    @timeout(1)
    def test_empty(self):
        xs = []
        even_before_odd(xs)
        self.assertFalse(xs)  # An empty list should remain empty

    @spectest(1)
    @timeout(1)
    def test_small(self):
        run_with_list(self, list(range(50)))

    @spectest(1)
    @timeout(1)
    def test_negative(self):
        run_with_list(self, list(range(-25, 20)))

    @spectest(5)
    @timeout(1)
    def test_many(self):
        random.seed(41652)
        for x in range(100, 500, 50):
            run_with_list(self, random.sample(range(-x, x), x))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
