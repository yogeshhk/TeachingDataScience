import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import more_vowels


class TestSolution(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_empty(self):
        self.assertTrue(more_vowels("") == 0)

    @yourtest
    @spectest(1)
    def test_single_letter(self):
        self.assertEqual(more_vowels("z"), -1)
        self.assertEqual(more_vowels("a"), 1)

    @yourtest
    @spectest(1)
    def test_example(self):
        self.assertEqual(more_vowels("aab"), 1)
        self.assertEqual(more_vowels("aaab"), 1)
        self.assertEqual(more_vowels("abb"), -1)
        self.assertEqual(more_vowels("abbb"), -1)
        self.assertEqual(more_vowels("aabb"), 0)

    @spectest(2)
    def test_larger_words(self):
        self.assertEqual(more_vowels("pineapple"), -1)
        self.assertEqual(more_vowels("anaconda"), 0)
        self.assertEqual(more_vowels("avocado"), 1)

    @spectest(2)
    def test_massive_words(self):
        self.assertEqual(more_vowels("abc" * 100), -1)
        self.assertEqual(more_vowels("opo" * 100), 1)
        self.assertEqual(more_vowels("momo" * 100), 0)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
