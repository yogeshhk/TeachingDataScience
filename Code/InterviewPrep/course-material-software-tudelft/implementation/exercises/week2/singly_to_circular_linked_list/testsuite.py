import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .solution import LL


class TestSolution(unittest.TestCase):

    # Tests for basic LL behaviour
    @yourtest
    @spectest(1)
    def test_small_all_add_remove(self):
        ll = LL()
        ll.add_last("last")
        ll.add_first("first")
        ll.add_at_position(1, "middle")
        self.assertEqual(ll.get_head(), "first")
        self.assertEqual(ll.remove_first(), "first")
        self.assertEqual(ll.get_tail(), "last")
        self.assertEqual(ll.get_head(), "middle")
        self.assertEqual(ll.remove_first(), "middle")
        self.assertEqual(ll.get_head(), "last")
        self.assertEqual(ll.get_tail(), "last")
        self.assertEqual(ll.remove_first(), "last")
        self.assertEqual(ll.get_head(), None)
        self.assertEqual(ll.get_tail(), None)

    @yourtest
    @spectest(1)
    def test_add_first(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        ll.add_first("three")
        self.assertEqual(ll.get_head(), "three")
        self.assertEqual(ll.get_tail(), "three")
        ll.add_first("two")
        self.assertEqual(ll.get_head(), "two")
        self.assertEqual(ll.get_tail(), "three")
        ll.add_first("one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.get_tail(), "three")

    @yourtest
    @spectest(1)
    def test_add_last(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        self.assertEqual(ll.get_tail(), None)
        ll.add_last("one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.get_tail(), "one")
        ll.add_last("two")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.get_tail(), "two")
        ll.add_last("three")
        self.assertEqual(ll.head.next_node.next_node.value, "three")
        self.assertEqual(ll.get_tail(), "three")

    @yourtest
    @spectest(1)
    def test_add_at_position(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        self.assertEqual(ll.get_tail(), None)
        ll.add_at_position(0, "one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.get_tail(), "one")
        ll.add_at_position(1, "two")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.get_tail(), "two")
        ll.add_at_position(2, "three")
        self.assertEqual(ll.head.next_node.next_node.value, "three")
        self.assertEqual(ll.get_tail(), "three")

    @yourtest
    @spectest(1)
    def test_remove_first(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.remove_first(), "one")
        self.assertEqual(ll.remove_first(), "two")
        self.assertEqual(ll.remove_first(), "three")
        self.assertEqual(ll.remove_first(), None)

    @yourtest
    @spectest(1)
    def test_remove_last(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.remove_last(), "three")
        self.assertEqual(ll.remove_last(), "two")
        self.assertEqual(ll.remove_last(), "one")
        self.assertEqual(ll.remove_last(), None)

    @yourtest
    @spectest(1)
    def test_remove_from_position(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.remove_from_position(2), "three")
        self.assertEqual(ll.remove_from_position(2), None)
        self.assertEqual(ll.remove_from_position(0), "one")
        self.assertEqual(ll.remove_from_position(1), None)
        self.assertEqual(ll.remove_from_position(0), "two")

    # Tests for CLL behaviour
    @yourtest
    @spectest(1)
    def test_cll_add_first(self):
        ll = LL()
        ll.add_first("last")
        ll.add_first("middle")
        ll.add_first("first")
        self.assertEqual(ll.head.value, "first")
        self.assertEqual(ll.head.next_node.value, "middle")
        self.assertEqual(ll.head.next_node.next_node.value, "last")
        self.assertEqual(ll.head.next_node.next_node.next_node.value, "first")
        self.assertEqual(ll.get_tail(), "last")

    @yourtest
    @spectest(1)
    def test_cll_remove_first(self):
        ll = LL()
        ll.add_first("last")
        ll.add_first("middle")
        ll.add_first("first")
        self.assertEqual(ll.remove_first(), "first")
        self.assertEqual(ll.head.value, "middle")
        self.assertEqual(ll.head.next_node.value, "last")
        self.assertEqual(ll.head.next_node.next_node.value, "middle")
        self.assertEqual(ll.remove_first(), "middle")
        self.assertEqual(ll.head.value, "last")
        self.assertEqual(ll.head.next_node.value, "last")
        self.assertEqual(ll.remove_first(), "last")
        self.assertEqual(ll.remove_first(), None)

    @yourtest
    @spectest(1)
    def test_cll_add_last(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        ll.add_last("one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.head.next_node.value, "one")
        ll.add_last("two")
        self.assertEqual(ll.head.value, "one")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.head.next_node.next_node.value, "one")
        ll.add_last("three")
        self.assertEqual(ll.head.value, "one")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.head.next_node.next_node.value, "three")
        self.assertEqual(ll.head.next_node.next_node.next_node.value, "one")

    @yourtest
    @spectest(1)
    def test_cll_remove_last(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.head.next_node.next_node.value, "three")
        self.assertEqual(ll.head.next_node.next_node.next_node.value, "one")
        self.assertEqual(ll.remove_last(), "three")
        self.assertEqual(ll.head.next_node.next_node.value, "one")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.remove_last(), "two")
        self.assertEqual(ll.head.next_node.value, "one")
        self.assertEqual(ll.head.value, "one")
        self.assertEqual(ll.remove_last(), "one")
        self.assertEqual(ll.head, None)
        self.assertEqual(ll.remove_last(), None)

    @yourtest
    @spectest(1)
    def test_cll_add_at_position(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        ll.add_at_position(0, "one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.head.next_node.value, "one")
        ll.add_at_position(1, "two")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.head.next_node.next_node.value, "one")
        ll.add_at_position(2, "three")
        self.assertEqual(ll.head.next_node.next_node.value, "three")
        self.assertEqual(ll.head.next_node.next_node.next_node.value, "one")

    @yourtest
    @spectest(1)
    def test_cll_remove_from_position(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.head.next_node.next_node.next_node.value, "one")
        self.assertEqual(ll.remove_from_position(2), "three")
        self.assertEqual(ll.remove_from_position(2), None)
        self.assertEqual(ll.head.next_node.next_node.value, "one")
        self.assertEqual(ll.remove_from_position(0), "one")
        self.assertEqual(ll.head.value, "two")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.remove_from_position(1), None)
        self.assertEqual(ll.remove_from_position(0), "two")


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
