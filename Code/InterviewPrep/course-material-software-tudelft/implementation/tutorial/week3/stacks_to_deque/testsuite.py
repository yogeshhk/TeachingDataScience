import unittest

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .library import Stack
from .solution import Deque


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_left(self):
        dq = Deque()
        for x in range(10):
            dq.add_first(x)
        for x in range(9, -1, -1):
            self.assertEqual(dq.remove_first(), x)

    @yourtest
    @spectest(1)
    def test_right(self):
        dq = Deque()
        for x in range(10):
            dq.add_last(x)
        for x in range(9, -1, -1):
            self.assertEqual(dq.remove_last(), x)

    @yourtest
    @spectest(1)
    def test_left_to_right(self):
        dq = Deque()
        for x in range(20):
            dq.add_first(x)
        for x in range(20):
            self.assertEqual(dq.remove_last(), x)

    @yourtest
    @spectest(1)
    def test_right_to_left(self):
        dq = Deque()
        for x in range(20):
            dq.add_last(x)
        for x in range(20):
            self.assertEqual(dq.remove_first(), x)

    @spectest(2)
    def test_mixed(self):
        dq = Deque()
        for x in range(10):
            dq.add_first(x)
            dq.add_last(10 + x)
        self.assertTrue(len(dq), 20)

        for x in range(9, -1, -1):
            self.assertEqual(dq.remove_first(), x)
            self.assertEqual(len(dq), x + 10)
        self.assertEqual(len(dq), 10)

        for x in range(10, 20):
            self.assertEqual(dq.remove_first(), x)
            self.assertEqual(len(dq), 19 - x)
        self.assertEqual(len(dq), 0)

    @spectest(2)
    def test_use_stacks_only(self):
        dq = Deque()
        for x in range(5):
            dq.add_first(x)
        self.assertEqual(dq.remove_last(), 0)
        self.assertEqual(dq.remove_first(), 4)

        self.assertEqual(len(dq), 3)
        self.assertEqual(len(dq.__dict__), 2)
        for _, v in dq.__dict__.items():
            self.assertIsInstance(v, Stack)

    @spectest(2)
    @timeout(1.2)
    def test_efficiency_1_stack(self):
        dq = Deque()
        for x in range(200000):
            dq.add_first(x)
        for x in range(199999, -1, -1):
            self.assertEqual(dq.remove_first(), x)
        for x in range(200000):
            dq.add_last(x)
        for x in range(199999, -1, -1):
            self.assertEqual(dq.remove_last(), x)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
