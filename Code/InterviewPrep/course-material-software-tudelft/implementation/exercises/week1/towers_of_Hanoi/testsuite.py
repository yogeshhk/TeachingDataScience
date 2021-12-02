import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .library import Peg
from .solution import hanoi


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_three_disks(self):
        A = Peg()
        A.push(3)
        A.push(2)
        A.push(1)
        B, C = Peg(), Peg()
        hanoi(3, A, B, C)
        self.assertEqual(C.pop(), 1)
        self.assertEqual(C.pop(), 2)
        self.assertEqual(C.pop(), 3)

    @spectest(5)
    def test_hanoi_small(self):
        A = Peg()
        for i in range(8, 0, -1):
            A.push(i)
        B, C = Peg(), Peg()
        hanoi(8, A, B, C)
        for i in range(1, 9):
            self.assertEqual(C.pop(), i)

    @spectest(5)
    def test_hanoi_medium(self):
        A = Peg()
        for i in range(13, 0, -1):
            A.push(i)
        B, C = Peg(), Peg()
        hanoi(13, A, B, C)
        for i in range(1, 14):
            self.assertEqual(C.pop(), i)

    @spectest(5)
    def test_hanoi_large(self):
        A = Peg()
        for i in range(16, 0, -1):
            A.push(i)
        B, C = Peg(), Peg()
        hanoi(16, A, B, C)
        for i in range(1, 17):
            self.assertEqual(C.pop(), i)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
