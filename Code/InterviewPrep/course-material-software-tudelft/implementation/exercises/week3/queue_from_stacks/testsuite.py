import unittest
import random

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import Queue
from .library import Stack


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        q = Queue()
        for x in range(10):
            q.enqueue(x)
        self.assertEqual(10, len(q))
        for x in range(10):
            self.assertEqual(x, q.first())
            self.assertFalse(q.is_empty())
            self.assertEqual(x, q.dequeue())
        self.assertTrue(q.is_empty())

    @spectest(1)
    def test_unordered(self):
        random.seed(41907)
        items = list(range(100))
        random.shuffle(items)
        q = Queue()
        for x in items:
            q.enqueue(x)
        for x in items:
            self.assertEqual(x, q.dequeue())
        self.assertTrue(q.is_empty())

    @spectest(2)
    def test_use_stacks_only(self):
        q = Queue()
        for x in range(5):
            q.enqueue(x)
        self.assertEqual(0, q.dequeue())
        self.assertEqual(4, len(q))
        self.assertEqual(2, len(q.__dict__))
        for _, v in q.__dict__.items():
            self.assertIsInstance(v, Stack)

    @spectest(2)
    def test_many(self):
        q = Queue()
        for x in range(900):
            q.enqueue(x)
        self.assertEqual(0, q.first())
        for x in range(900):
            self.assertEqual(x, q.dequeue())
        self.assertTrue(q.is_empty())

    @spectest(1)
    def test_first(self):
        q = Queue()
        q.enqueue("first")
        q.enqueue("second")
        self.assertFalse(q.is_empty())
        self.assertEqual("first", q.first())
        self.assertEqual(2, len(q))
        self.assertEqual("first", q.dequeue())
        self.assertEqual(1, len(q))
        self.assertFalse(q.is_empty())
        self.assertEqual("second", q.first())
        self.assertEqual(1, len(q))
        self.assertEqual("second", q.dequeue())
        self.assertEqual(0, len(q))
        self.assertTrue(q.is_empty())


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
