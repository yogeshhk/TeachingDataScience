import unittest

from decorators import spectest, yourtest, timeout, solution_only
from weblabTestRunner import TestRunner
from .solution import RaysWallet

# A big wallet used for testing efficiency, created outside
# to avoid interference with the run time.
if solution_only:
    big = RaysWallet()
    big.notes = list(range(50000000, 1, -1))


class TestSuite(unittest.TestCase):
    @yourtest
    @spectest(1)
    def test_add(self):
        w = RaysWallet()
        w.add(5)
        self.assertEqual(w.notes, [5])

    @yourtest
    @spectest(1)
    def test_remove(self):
        w = RaysWallet()
        w.add(5)
        w.remove(5)
        self.assertTrue(w.is_empty())

    @yourtest
    @spectest(1)
    def test_contains(self):
        w = RaysWallet()
        w.add(5)
        self.assertTrue(w.contains(5))

    @yourtest
    @spectest(1)
    def test_remove_mtf(self):
        w = RaysWallet()
        w.add(5)
        w.add(5)
        w.add(10)
        w.remove_mtf(5)
        self.assertEqual(w.notes, [5, 10])

    @spectest(1)
    @timeout(0.1)
    def test_add_increasing(self):
        w = RaysWallet()
        w.add(5)
        w.add(10)
        w.add(20)
        self.assertEqual(w.notes, [20, 10, 5])

    @spectest(1)
    @timeout(1)
    def test_add_decreasing(self):
        w = RaysWallet()
        w.add(20)
        w.add(10)
        w.add(5)
        self.assertEqual(w.notes, [20, 10, 5])

    @spectest(1)
    @timeout(0.1)
    def test_add_random(self):
        w = RaysWallet()
        w.add(10)
        w.add(5)
        w.add(20)
        self.assertEqual(w.notes, [20, 10, 5])

    @spectest(1)
    def test_remove_same(self):
        w = RaysWallet()
        w.add(5)
        w.add(5)
        w.add(5)
        w.remove(5)
        self.assertEqual(w.notes, [5, 5])

    @spectest(1)
    def test_remove_not_existing(self):
        w = RaysWallet()
        w.add(5)
        w.add(5)
        w.add(5)
        w.remove(10)
        self.assertEqual(w.notes, [5, 5, 5])

    @spectest(1)
    def test_remove_empty(self):
        w = RaysWallet()
        self.assertEqual(w.notes, [])

    @spectest(1)
    @timeout(0.1)
    def test_contains(self):
        w = RaysWallet()
        w.add(5)
        w.add(5)
        w.add(5)
        self.assertTrue(w.contains(9))

    @spectest(1)
    @timeout(0.1)
    def test_does_not_contain(self):
        w = RaysWallet()
        w.add(5)
        w.add(5)
        w.add(5)
        self.assertFalse(w.contains(23))

    @spectest(1)
    @timeout(0.1)
    def test_contains_empty(self):
        w = RaysWallet()
        self.assertFalse(w.contains(23))

    @spectest(1)
    @timeout(0.1)
    def test_contains_zero(self):
        w = RaysWallet()
        w.add(5)
        w.add(5)
        w.add(5)
        self.assertTrue(w.contains(0))

    @spectest(1)
    @timeout(0.1)
    def test_contains_efficient(self):
        self.assertTrue(big.contains(1000))

    @spectest(1)
    @timeout(0.1)
    def test_remove_mtf_not_existing(self):
        w = RaysWallet()
        w.add(10)
        w.add(5)
        w.add(5)
        w.remove(15)
        self.assertEqual(w.notes, [10, 5, 5])

    @spectest(1)
    @timeout(0.1)
    def test_remove_mft_empty(self):
        w = RaysWallet()
        self.assertEqual(w.notes, [])

    @spectest(1)
    @timeout(0.1)
    def test_remove_mft_last(self):
        w = RaysWallet()
        w.add(20)
        w.add(10)
        w.add(5)
        w.remove_mtf(5)
        self.assertEqual(w.notes, [20, 10])

    @spectest(1)
    @timeout(0.1)
    def test_remove_mft_first(self):
        w = RaysWallet()
        w.add(20)
        w.add(10)
        w.add(5)
        w.remove_mtf(20)
        self.assertEqual(w.notes, [10, 5])

    @spectest(1)
    @timeout(0.1)
    def test_remove_mft_proper(self):
        w = RaysWallet()
        w.add(50)
        w.add(20)
        w.add(10)
        w.add(10)
        w.add(10)
        w.add(10)
        w.add(5)
        w.add(5)
        w.remove_mtf(10)
        self.assertEqual(w.notes, [10, 10, 10, 50, 20, 5, 5])


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
