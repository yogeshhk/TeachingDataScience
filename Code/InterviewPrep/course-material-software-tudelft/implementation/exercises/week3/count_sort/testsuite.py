import unittest
import random
from decorators import spectest, yourtest, timeout, remove
from weblabTestRunner import TestRunner
from .solution import count_sort


class TestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        random.seed(78192)
        cls.massive_list = [random.randint(0, 5000) for _ in range(1_000_000)]
        cls.massive_list_sorted = sorted(cls.massive_list)

    @spectest(1)
    @yourtest
    @timeout(0.1)
    def test_duplicates(self):
        random.seed(42)
        xs = [random.randint(0, 20) for _ in range(200)]
        self.assertEqual(count_sort(xs), sorted(xs))

    @spectest(1)
    @yourtest
    @timeout(0.1)
    def test_uniques(self):
        random.seed(42)
        xs = list(range(100))
        random.shuffle(xs)
        self.assertEqual(count_sort(xs), list(range(100)))

    @spectest(3)
    @yourtest
    @timeout(3)  # TODO change this on weblab to 1.5s
    # Tests that this is a fast implementation, only allows 1.5s of run time to sort 1M items.
    # The list is long, but contains a relatively small range of values.
    # This is where count sort shines.
    # For the reference: we tested this locally and count_sort completes this test in around 0.25 seconds,
    # while merge sort takes around 10 seconds to complete on the same machine.
    # Increasing the list size to 10M items leads to a runtime of <2.5s for count sort,
    # while merge sort takes over 2 minutes!
    def test_efficiency(self):
        self.assertEqual(count_sort(self.massive_list), self.massive_list_sorted)

    @spectest(1)
    @timeout(0.2)
    def test_larger_range(self):
        random.seed(59175)
        xs = [random.randint(-5000, 5000) for _ in range(10000)]
        self.assertEqual(count_sort(xs), sorted(xs))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
