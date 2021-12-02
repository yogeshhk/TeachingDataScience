import unittest
from typing import List

from decorators import spectest, yourtest, remove
from weblabTestRunner import TestRunner
from .solution import PuzzleSolver


class Tester:
    def __init__(self, a: str, b: str, c: str):
        self.a = a
        self.b = b
        self.c = c
        self.letters = sorted(set(list(self.a) + list(self.b) + list(self.c)))

    # Creates a number from the given string based on the given list
    def string_to_num(self, item: str, xs: List[int]) -> int:
        return int(''.join([str(xs[self.letters.index(char)]) for char in item]))

    # Tests if a given permutation is a correct solution
    def test(self, xs: List[int]) -> bool:
        if len(xs) != len(self.letters):
            return False
        a_num = self.string_to_num(self.a, xs)
        b_num = self.string_to_num(self.b, xs)
        c_num = self.string_to_num(self.c, xs)
        return a_num + b_num == c_num


class TestSolution(unittest.TestCase):

    @yourtest
    def test_example_1(self):
        tester = Tester("pot", "pan", "bib")
        solver = PuzzleSolver("pot", "pan", "bib")
        sol = solver.solve()
        self.assertTrue(tester.test(sol))

    @spectest(2)
    @yourtest
    def test_example_2(self):
        tester = Tester("dog", "cat", "pig")
        solver = PuzzleSolver("dog", "cat", "pig")
        sol = solver.solve()
        self.assertTrue(tester.test(sol))

    @spectest(3)
    def test_example_3(self):
        tester = Tester("boy", "girl", "baby")
        solver = PuzzleSolver("boy", "girl", "baby")
        sol = solver.solve()
        self.assertTrue(tester.test(sol))

    @remove
    def test_4(self):
        tester = Tester("this", "is", "four")
        solver = PuzzleSolver("this", "is", "four")
        sol = solver.solve()
        self.assertTrue(tester.test(sol))

    @spectest(1)
    def test_small(self):
        tester = Tester("a", "b", "c")
        solver = PuzzleSolver("a", "b", "c")
        sol = solver.solve()
        self.assertTrue(tester.test(sol))

    @remove
    def test_medium(self):
        tester = Tester("zxy", "uwv", "ubw")
        solver = PuzzleSolver("zxy", "uwv", "ubw")
        sol = solver.solve()
        self.assertTrue(tester.test(sol))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
