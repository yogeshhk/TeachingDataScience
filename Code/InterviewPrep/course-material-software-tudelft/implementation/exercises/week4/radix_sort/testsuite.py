import random
import unittest

from decorators import spectest, timeout, yourtest
from weblabTestRunner import TestRunner
from .solution import radix_sort_lsd


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_small(self):
        xs = ["0622222222", "0611111111", "0633333333"]
        xs = radix_sort_lsd(xs)
        self.assertEqual(xs, ["0611111111", "0622222222", "0633333333"])

    @spectest(1)
    def test_empty(self):
        xs = []
        xs = radix_sort_lsd(xs)
        self.assertEqual(xs, [])

    @spectest(1)
    def test_one(self):
        xs = ["0623567846"]
        xs = radix_sort_lsd(xs)
        self.assertEqual(xs, ["0623567846"])

    @spectest(1)
    def test_sorted(self):
        xs = ["0611111111", "0622222222", "0633333333", "0644444444"]
        xs = radix_sort_lsd(xs)
        self.assertEqual(xs, ["0611111111", "0622222222", "0633333333", "0644444444"])

    @spectest(1)
    def test_reverse(self):
        xs = ["0699999999", "0688888888", "0677777777", "0666666666"]
        xs = radix_sort_lsd(xs)
        self.assertEqual(xs, ["0666666666", "0677777777", "0688888888", "0699999999"])

    @spectest(1)
    def test_equal_numbers(self):
        xs = ["0682935734" for _ in range(42)]
        xs = radix_sort_lsd(xs)
        self.assertEqual(xs, ["0682935734" for _ in range(42)])

    @spectest(5)
    def test_small(self):
        random.seed(385762)
        # {number:08d} formats `number` with up to 8 leading 0s
        data = [f"06{random.randint(0, 1_0000_0000):08d}" for _ in range(40)]
        data2 = sorted(data)
        self.assertEqual(data2, radix_sort_lsd(data))

    @spectest(5)
    @timeout(2)
    def test_large(self):
        random.seed(52364)
        for i in range(16):
            size = 2 ** i
            data = [f"06{random.randint(0, 1_0000_0000):08d}" for _ in range(size)]
            data2 = sorted(data)
            self.assertEqual(data2, radix_sort_lsd(data))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
