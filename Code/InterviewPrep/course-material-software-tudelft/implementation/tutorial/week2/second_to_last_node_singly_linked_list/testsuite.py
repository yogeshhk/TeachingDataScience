import unittest

from decorators import spectest, timeout, yourtest
from .solution import SLL
from weblabTestRunner import TestRunner


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_one_element(self):
        sll = SLL()
        sll.add_first("Manager")
        self.assertEqual(sll.second_to_last_node(), None)

    @yourtest
    @spectest(1)
    def test_two_elements(self):
        sll = SLL()
        sll.add_first("Tortoise")
        sll.add_first("Hare")
        self.assertEqual(sll.second_to_last_node(), "Hare")

    @yourtest
    @spectest(1)
    def test_some_elements(self):
        sll = SLL()
        sll.add_first("Mary")
        sll.add_first("Dave")
        sll.add_first("Elsie")
        sll.add_first("Oscar")
        sll.add_first("Suzie")
        sll.add_first("Billy")
        self.assertEqual(sll.second_to_last_node(), "Dave")


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
