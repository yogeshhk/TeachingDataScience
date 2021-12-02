import random
import unittest

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import first_index_of, all_indices_of


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_empty(self):
        test_list = []
        self.assertEqual(None, first_index_of(42, test_list))

    @yourtest
    @spectest(1)
    def test_single_element(self):
        test_list = [1]
        self.assertEqual(0, first_index_of(1, test_list))

    @yourtest
    @spectest(1)
    def test_first_index_near_end(self):
        test_list = [42, 32, 34, 543, 532, 4234, 100, 9123]
        self.assertEqual(6, first_index_of(100, test_list))

    @yourtest
    @spectest(1)
    def test_small_list_duplicates(self):
        test_list = [1, 2, 3, 1, 2, 3]
        self.assertEqual([0, 3], list(all_indices_of(1, test_list)))
        self.assertEqual([1, 4], list(all_indices_of(2, test_list)))
        self.assertEqual([2, 5], list(all_indices_of(3, test_list)))

    @spectest(1)
    def test_only_duplicates(self):
        test_list = [1] * 100
        self.assertEqual(list(range(100)), list(all_indices_of(1, test_list)))

    @spectest(2)
    @timeout(1)
    def test_no_occurrence_first_index(self):
        test_list = [2, 3, 12, 5, 546, 42, 42, 42, 12, 3, 13]
        self.assertEqual(None, first_index_of(0, test_list))

    @spectest(2)
    @timeout(1)
    def test_no_occurrence_all_indices(self):
        test_list = [2, 3, 12, 5, 546, 42, 42, 42, 12, 3, 13]
        self.assertEqual([], list(all_indices_of(0, test_list)))

    @spectest(2)
    @timeout(1)
    def test_large_first_index(self):
        random.seed(101)
        # [54, 76, 64, 71, 45, 74, 55, 56, 60, 73, 55, 63, 70, 46, 58, 54, 52, 47, 70, 76]
        test_list = [random.choice(range(42, 77)) for _ in range(20)]
        self.assertEqual(1, first_index_of(76, test_list))
        self.assertEqual(12, first_index_of(70, test_list))
        self.assertEqual(3, first_index_of(71, test_list))
        self.assertEqual(6, first_index_of(55, test_list))

    @spectest(2)
    @timeout(1)
    def test_large_random_all_indices(self):
        random.seed(42)
        # [1, 1, 3, 2, 2, 2, 1, 1, 4, 1, 1, 1, 2, 2, 1, 2, 4, 2, 4, 3]
        test_list = [random.choice(range(1, 5)) for _ in range(20)]
        self.assertEqual([i for i, s in enumerate(test_list) if s == 1], list(all_indices_of(1, test_list)))
        self.assertEqual([i for i, s in enumerate(test_list) if s == 2], list(all_indices_of(2, test_list)))
        self.assertEqual([i for i, s in enumerate(test_list) if s == 3], list(all_indices_of(3, test_list)))
        self.assertEqual([i for i, s in enumerate(test_list) if s == 4], list(all_indices_of(4, test_list)))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
