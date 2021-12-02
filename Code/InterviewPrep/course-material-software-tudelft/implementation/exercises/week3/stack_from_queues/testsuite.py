import unittest
import random

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import Stack
from .library import Queue


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        s = Stack()
        for x in range(10):
            s.push(x)
        self.assertEqual(10, len(s))
        for x in reversed(range(10)):
            self.assertEqual(x, s.top())
            self.assertFalse(s.is_empty())
            self.assertEqual(x, s.pop())
        self.assertTrue(s.is_empty())

    @spectest(1)
    def test_unordered(self):
        random.seed(48193)
        items = list(range(100))
        random.shuffle(items)
        s = Stack()
        for x in items:
            s.push(x)
        for x in reversed(items):
            self.assertEqual(x, s.pop())
        self.assertTrue(s.is_empty())

    @spectest(2)
    def test_use_queues_only(self):
        s = Stack()
        for x in range(5):
            s.push(x)
        self.assertEqual(4, s.pop())
        self.assertEqual(4, len(s))
        self.assertEqual(2, len(s.__dict__))
        for _, v in s.__dict__.items():
            self.assertIsInstance(v, Queue)

    @spectest(2)
    def test_many_push_pop(self):
        s = Stack()
        for x in range(900):
            s.push(x)
        self.assertEqual(899, s.top())
        for x in reversed(range(900)):
            self.assertEqual(x, s.pop())
        self.assertTrue(s.is_empty())

    @spectest(1)
    def test_top(self):
        s = Stack()
        s.push("first")
        s.push("second")
        self.assertFalse(s.is_empty())
        self.assertEqual("second", s.top())
        self.assertEqual(2, len(s))
        self.assertEqual("second", s.pop())
        self.assertEqual(1, len(s))
        self.assertFalse(s.is_empty())
        self.assertEqual("first", s.top())
        self.assertEqual(1, len(s))
        self.assertEqual("first", s.pop())
        self.assertEqual(0, len(s))
        self.assertTrue(s.is_empty())


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
