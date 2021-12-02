import itertools
import random
import unittest

from decorators import spectest, yourtest, parameterized, remove
from weblabTestRunner import TestRunner

from .solution import simple_hash


@remove
def simple_hash_solution(string: str, s: int) -> int:
    a = 1003
    m = 1_000_000_009
    b = 0
    for char in string:
        b = (ord(char) + b * a) % m
    return b % s


@remove
def generate_a_heck_load_of_strings(solution):
    random.seed(421337)
    return [
        (s, s, solution(s, 1000000)) for s in (
            "".join(chr(random.choice(list(itertools.chain(range(48, 58), range(65, 91), range(97, 123)))))
                    for _ in range(random.randint(1, 1000)))
            for _ in range(996)  # Generating 1000 spec-tests in total
        )
    ]


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_simple_hash_empty(self):
        self.assertEqual(0, simple_hash("", 100))

    @yourtest
    @spectest(1)
    def test_simple_hash_one_character(self):
        # ord("x") = 120, modulo 100 = 20
        self.assertEqual(20, simple_hash("x", 100))

    @spectest(1)
    def test_simple_hash_two_characters(self):
        self.assertEqual(121 + 1003 * 120, simple_hash("xy", 1000000))

    @spectest(1)
    def test_simple_hash_three_characters(self):
        # ord("a") = 97
        self.assertEqual((99 + 1003 * (98 + 1003 * 97)) % 1000000, simple_hash("abc", 1000000))

    @parameterized(spectest, generate_a_heck_load_of_strings(simple_hash_solution))
    def test_simple_hash(self, i: str, o: int):
        self.assertEqual(o, simple_hash(i, 1000000))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
