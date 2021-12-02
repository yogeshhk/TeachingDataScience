import itertools
import random
import unittest

from decorators import spectest, yourtest, parameterized, remove
from weblabTestRunner import TestRunner

from .solution import hash_gnu_cc1, hash_gnu_cpp, hash_python_hash


@remove
def hash_gnu_cc1_solution(string: str, s: int) -> int:
    b = len(string)
    for char in string:
        b = ord(char) ^ b * 33
    return b % s


@remove
def hash_gnu_cpp_solution(string: str, s: int) -> int:
    b = 0
    for char in string:
        b = ord(char) + b * 4
    return b % s


@remove
def hash_python_hash_solution(string: str, s: int) -> int:
    return abs(hash(string)) % s


@remove
def generate_a_heck_load_of_strings(solution):
    random.seed(421337)
    return [
        (s, s, solution(s, 1000000)) for s in (
            "".join(chr(random.choice(list(itertools.chain(range(48, 58), range(65, 91), range(97, 123)))))
                    for _ in range(random.randint(1, 100)))
            for _ in range(496)  # Generating 1000 spec-tests in total
        )
    ]


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_hash_gnu_cc1_empty(self):
        self.assertEqual(0, hash_gnu_cc1("", 100))

    @yourtest
    @spectest(1)
    def test_hash_gnu_cc1_one_character(self):
        # ord("x") = 120, modulo 100 = 20
        self.assertEqual(120 ^ 33, hash_gnu_cc1("x", 100))

    @spectest(1)
    def test_hash_gnu_cc1_two_characters(self):
        self.assertEqual(121 ^ ((120 ^ 66) * 33), hash_gnu_cc1("xy", 1000000))

    @spectest(1)
    def test_hash_gnu_cc1_three_characters(self):
        # ord("a") = 97
        self.assertEqual(99 ^ ((98 ^ ((97 ^ 99) * 33)) * 33), hash_gnu_cc1("abc", 1000000))

    @remove  # Workaround because parameterized spec-tests are not removed yet by the generator
    @parameterized(spectest, generate_a_heck_load_of_strings(hash_gnu_cc1_solution))
    def test_hash_gnu_cc1(self, i: str, o: int):
        self.assertEqual(o, hash_gnu_cc1(i, 1000000))

    @yourtest
    @spectest(1)
    def test_hash_gnu_cpp_empty(self):
        self.assertEqual(0, hash_gnu_cpp("", 100))

    @yourtest
    @spectest(1)
    def test_hash_gnu_cpp_one_character(self):
        # ord("x") = 120, modulo 100 = 20
        self.assertEqual(20, hash_gnu_cpp("x", 100))

    @spectest(1)
    def test_hash_gnu_cpp_two_characters(self):
        self.assertEqual(121 + 120 * 4, hash_gnu_cpp("xy", 1000000))

    @spectest(1)
    def test_hash_gnu_cpp_three_characters(self):
        # ord("a") = 97
        self.assertEqual(99 + (98 + 97 * 4) * 4, hash_gnu_cpp("abc", 1000000))

    @remove
    @parameterized(spectest, generate_a_heck_load_of_strings(hash_gnu_cpp_solution))
    def test_hash_gnu_cpp(self, i: str, o: int):
        self.assertEqual(o, hash_gnu_cpp(i, 1000000))

    @yourtest
    @spectest(1)
    def test_hash_python_hash_empty(self):
        self.assertEqual(0, hash_python_hash("", 100))

    @spectest(1)
    def test_hash_python_hash_one_character(self):
        self.assertEqual(hash_python_hash_solution("x", 100), hash_python_hash("x", 100))

    @spectest(1)
    def test_hash_python_hash_two_characters(self):
        self.assertEqual(hash_python_hash_solution("xy", 1000000), hash_python_hash("xy", 1000000))

    @spectest(1)
    def test_hash_python_hash_three_characters(self):
        self.assertEqual(hash_python_hash_solution("abc", 1000000), hash_python_hash("abc", 1000000))

    @remove
    @parameterized(spectest, generate_a_heck_load_of_strings(hash_python_hash_solution))
    def test_hash_python_hash(self, i: str, o: int):
        self.assertEqual(o, hash_python_hash(i, 1000000))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
