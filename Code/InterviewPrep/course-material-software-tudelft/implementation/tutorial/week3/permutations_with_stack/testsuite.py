import itertools
import unittest

from decorators import spectest, yourtest, timeout
from .solution import permutations
from weblabTestRunner import TestRunner


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    @timeout(0.1)
    def test_small(self):
        ingredients = ["cheese", "lettuce", "burger"]
        result = permutations(ingredients)
        self.assertEqual(len(result), 6)
        self.assertEqual(result, set(itertools.permutations(ingredients)))

    @yourtest
    @spectest(1)
    @timeout(0.1)
    def test_duplicates(self):
        ingredients = ["burger", "burger", "cheese"]
        result = permutations(ingredients)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, set(itertools.permutations(ingredients)))

    @spectest(1)
    @timeout(1)
    def test_large(self):
        ingredients = ["lettuce", "top bun", "bottom bun", "burger", "pickle", "ketchup"]
        result = permutations(ingredients)
        self.assertEqual(len(result), 720)
        self.assertEqual(result, set(itertools.permutations(ingredients)))

    @spectest(1)
    @timeout(1)
    def test_large_duplicates(self):
        ingredients = ["lettuce", "tomato", "bun", "bun", "cheese", "burger", "bacon"]
        result = permutations(ingredients)
        self.assertEqual(len(result), 2520)
        self.assertEqual(result, set(itertools.permutations(ingredients)))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
