import random
import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import find_keys


class TestSuite(unittest.TestCase):

    @spectest(1)
    @yourtest
    def test_one_list(self):
        test_dict = {
            "magic": {42}
        }
        self.assertEqual({"magic"}, find_keys(42, test_dict))
        self.assertEqual(set(), find_keys(41, test_dict))

    @yourtest
    @spectest(1)
    def test_example(self):
        test_dict = {
            "x": {1, 2, 3},
            "y": {2, 4},
            "h": {4, 5}
        }
        self.assertEqual({"x"}, find_keys(1, test_dict))
        self.assertEqual({"y", "h"}, find_keys(4, test_dict))

    @spectest(1)
    def test_same_sets(self):
        test_dict = {
            "a": {1, 2},
            "b": {1, 2},
            "c": {1, 2},
            "d": {1, 2},
            "e": {1, 2},
            "f": {1, 2},
            "g": {1, 2},
            "h": {1, 2},
            "i": {1, 2},
            "j": {1, 2},
        }
        self.assertEqual(set(), find_keys(0, test_dict))
        self.assertEqual({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}, find_keys(1, test_dict))
        self.assertEqual({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}, find_keys(2, test_dict))

    @spectest(1)
    def test_random(self):
        random.seed(42)
        test_dict = {}
        for val in range(42, 52):
            test_dict[''.join(random.choice('abc') for _ in range(5))] = {val}
        for val in range(42, 52):
            key = next(iter(find_keys(val, test_dict)))
            self.assertEqual(test_dict[key], {val})

    @spectest(3)
    def test_large(self):
        test_dict = {
            "anaconda": {11, 1, 27, 596},
            "bear": {13, 9, 15, 21, 4, 8},
            "crab": {7, 4, 18, 14, 19, 5},
            "duck": {5, 12, 6, 18, 9, 8, 14, 7, 0, 13, 1},
            "eagle": {518, 1017, 757, 10011, 537},
            "fox": {150, 140, 101, 890, 96},
            "giraffe": {1337, 53, 42, 45},
            "horse": {0, 1},
            "iguana": {42},
            "jackal": {7, 77},
        }
        self.assertEqual(set(), find_keys(2, test_dict))
        self.assertEqual({"crab", "duck", "jackal"}, find_keys(7, test_dict))
        self.assertEqual({"giraffe"}, find_keys(1337, test_dict))
        self.assertEqual({"giraffe", "iguana"}, find_keys(42, test_dict))
        self.assertEqual({"anaconda"}, find_keys(596, test_dict))
        self.assertEqual({"fox"}, find_keys(101, test_dict))
        self.assertEqual({"duck", "horse"}, find_keys(0, test_dict))
        self.assertEqual({"anaconda", "duck", "horse"}, find_keys(1, test_dict))
        self.assertEqual({"eagle"}, find_keys(10011, test_dict))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
