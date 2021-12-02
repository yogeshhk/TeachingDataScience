import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import primes_in_range, count_primes, is_prime


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_is_prime_true(self):
        self.assertTrue(is_prime(7))

    @yourtest
    @spectest(1)
    def test_is_prime_false(self):
        self.assertFalse(is_prime(8))

    @yourtest
    @spectest(1)
    def test_primes_in_range(self):
        self.assertEqual(primes_in_range(2, 6), [2, 3, 5])

    @yourtest
    @spectest(1)
    def test_count_primes(self):
        self.assertEqual(count_primes(2, 6), 3)

    @spectest(1)
    def test_is_prime_one(self):
        self.assertFalse(is_prime(1))

    @spectest(1)
    def test_is_prime_negative(self):
        self.assertFalse(is_prime(-42))

    @spectest(1)
    def test_is_prime_true_big(self):
        self.assertTrue(is_prime(23))

    @spectest(1)
    def test_is_prime_false_big(self):
        self.assertFalse(is_prime(42))

    @spectest(1)
    def test_primes_in_range_none(self):
        self.assertEqual(primes_in_range(20, 22), [])

    @spectest(1)
    def test_primes_in_range_negative(self):
        self.assertEqual(primes_in_range(-5, 4), [2, 3])

    @spectest(1)
    def test_count_primes_none(self):
        self.assertEqual(count_primes(20, 22), 0)

    @spectest(1)
    def test_count_primes_negative(self):
        self.assertEqual(count_primes(-5, 4), 2)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
