import unittest

from decorators import spectest, yourtest, parameterized
from weblabTestRunner import TestRunner
from .solution import celsius_to_fahrenheit, armstrong_number_in_interval


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_celsius_to_fahrenheit(self):
        self.assertEqual(celsius_to_fahrenheit(1), 33.8)

    @parameterized(spectest, [
        ("zero", 0, 32),
        ("negative", -23, -9.4),
        ("float", 4.2, 39.6)
    ])
    def test_celsius_to_fahrenheit_parameterized(self, i: float, o: float):
        self.assertEqual(celsius_to_fahrenheit(i), o)

    @yourtest
    @spectest(1)
    def test_armstrong_numbers(self):
        self.assertTrue(armstrong_number_in_interval(150, 160))

    @spectest(1)
    def test_armstrong_numbers_true(self):
        self.assertTrue(armstrong_number_in_interval(1600, 1700))

    @spectest(1)
    def test_armstrong_number_false(self):
        self.assertFalse(armstrong_number_in_interval(20, 30))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
