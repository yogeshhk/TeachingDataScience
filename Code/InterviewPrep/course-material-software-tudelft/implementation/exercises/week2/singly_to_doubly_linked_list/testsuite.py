import unittest

from decorators import yourtest, spectest, solution_only, timeout
from .solution import LL
from weblabTestRunner import TestRunner

if solution_only:
    huge_linked_list = LL()
    for x in range(100_000):
        huge_linked_list.add_first(str(x))


class TestSuite(unittest.TestCase):

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
        self.assertEqual(ll.get_head(), "middle")
        self.assertEqual(ll.remove_at_position(1), "last")
        self.assertEqual(ll.get_head(), "middle")
        self.assertEqual(ll.remove_last(), "middle")
        self.assertEqual(ll.get_head(), None)

    @yourtest
    @spectest(1)
    def test_add_first(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        ll.add_first("three")
        self.assertEqual(ll.get_head(), "three")
        ll.add_first("two")
        self.assertEqual(ll.get_head(), "two")
        ll.add_first("one")
        self.assertEqual(ll.get_head(), "one")

    @yourtest
    @spectest(1)
    def test_add_last(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        ll.add_last("one")
        self.assertEqual(ll.get_head(), "one")
        ll.add_last("two")
        self.assertEqual(ll.head.next_node.value, "two")
        ll.add_last("three")
        self.assertEqual(ll.head.next_node.next_node.value, "three")

    @yourtest
    @spectest(1)
    def test_add_at_position(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        ll.add_at_position(0, "one")
        self.assertEqual(ll.get_head(), "one")
        ll.add_at_position(1, "two")
        self.assertEqual(ll.head.next_node.value, "two")
        ll.add_at_position(2, "three")
        self.assertEqual(ll.head.next_node.next_node.value, "three")

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
    def test_remove_at_position(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.remove_at_position(2), "three")
        self.assertEqual(ll.remove_at_position(2), None)
        self.assertEqual(ll.remove_at_position(0), "one")
        self.assertEqual(ll.remove_at_position(1), None)
        self.assertEqual(ll.remove_at_position(0), "two")

    # Tests for DLL behaviour
    @yourtest
    @spectest(1)
    def test_doubly_linked_list(self):
        ll = LL()
        ll.add_last("last")
        ll.add_first("first")
        ll.add_at_position(1, "middle")
        self.assertEqual(ll.get_head(), "first")
        self.assertEqual(ll.get_tail(), "last")
        self.assertEqual(ll.remove_first(), "first")
        self.assertEqual(ll.get_tail(), "last")
        self.assertEqual(ll.tail.prev_node.value, "middle")
        self.assertEqual(ll.get_head(), "middle")
        self.assertEqual(ll.head.next_node.value, "last")
        self.assertEqual(ll.remove_at_position(1), "last")
        self.assertEqual(ll.get_tail(), "middle")
        self.assertEqual(ll.get_head(), "middle")
        self.assertEqual(ll.remove_last(), "middle")
        self.assertEqual(ll.get_head(), None)

    @spectest(1)
    def test_dll_add_first(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        ll.add_first("three")
        self.assertEqual(ll.get_head(), "three")
        self.assertEqual(ll.get_tail(), "three")
        ll.add_first("two")
        self.assertEqual(ll.get_head(), "two")
        self.assertEqual(ll.get_tail(), "three")
        self.assertEqual(ll.tail.prev_node.value, "two")
        ll.add_first("one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.get_tail(), "three")
        self.assertEqual(ll.tail.prev_node.value, "two")

    @spectest(1)
    def test_dll_add_last(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        self.assertEqual(ll.get_tail(), None)
        ll.add_last("one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.get_tail(), "one")
        ll.add_last("two")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.tail.prev_node.value, "one")
        self.assertEqual(ll.get_tail(), "two")
        ll.add_last("three")
        self.assertEqual(ll.head.next_node.next_node.value, "three")
        self.assertEqual(ll.tail.prev_node.value, "two")
        self.assertEqual(ll.get_tail(), "three")

    @spectest(1)
    def test_dll_add_at_position(self):
        ll = LL()
        self.assertEqual(ll.get_head(), None)
        self.assertEqual(ll.get_tail(), None)
        ll.add_at_position(0, "one")
        self.assertEqual(ll.get_head(), "one")
        self.assertEqual(ll.get_tail(), "one")
        ll.add_at_position(1, "two")
        self.assertEqual(ll.head.next_node.value, "two")
        self.assertEqual(ll.get_tail(), "two")
        self.assertEqual(ll.tail.prev_node.value, "one")
        ll.add_at_position(2, "three")
        self.assertEqual(ll.head.next_node.next_node.value, "three")
        self.assertEqual(ll.get_tail(), "three")
        self.assertEqual(ll.tail.prev_node.value, "two")
        ll.add_at_position(42, "four")
        self.assertEqual(ll.head.next_node.next_node.next_node.value, "four")
        self.assertEqual(ll.get_tail(), "four")
        self.assertEqual(ll.tail.prev_node.value, "three")

    @spectest(1)
    def test_dll_remove_first(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.remove_first(), "one")
        self.assertEqual(ll.get_head(), "two")
        self.assertEqual(ll.head.prev_node, None)
        self.assertEqual(ll.remove_first(), "two")
        self.assertEqual(ll.get_head(), "three")
        self.assertEqual(ll.head.prev_node, None)
        self.assertEqual(ll.remove_first(), "three")
        self.assertEqual(ll.get_head(), None)
        self.assertEqual(ll.remove_first(), None)

    @spectest(1)
    def test_dll_remove_last(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.remove_last(), "three")
        self.assertEqual(ll.get_tail(), "two")
        self.assertEqual(ll.tail.next_node, None)
        self.assertEqual(ll.remove_last(), "two")
        self.assertEqual(ll.get_tail(), "one")
        self.assertEqual(ll.tail.next_node, None)
        self.assertEqual(ll.remove_last(), "one")
        self.assertEqual(ll.get_tail(), None)
        self.assertEqual(ll.remove_last(), None)

    @spectest(1)
    def test_dll_remove_at_position(self):
        ll = LL()
        ll.add_first("three")
        ll.add_first("two")
        ll.add_first("one")
        self.assertEqual(ll.remove_at_position(2), "three")
        self.assertEqual(ll.get_tail(), "two")
        self.assertEqual(ll.tail.next_node, None)
        self.assertEqual(ll.tail.prev_node.value, "one")
        self.assertEqual(ll.remove_at_position(2), None)
        # Nothing should have changed
        self.assertEqual(ll.get_tail(), "two")
        self.assertEqual(ll.tail.next_node, None)
        self.assertEqual(ll.tail.prev_node.value, "one")
        self.assertEqual(ll.remove_at_position(0), "one")
        self.assertEqual(ll.get_head(), "two")
        self.assertEqual(ll.get_tail(), "two")
        self.assertEqual(ll.head.next_node, None)
        self.assertEqual(ll.head.prev_node, None)
        self.assertEqual(ll.remove_at_position(1), None)
        self.assertEqual(ll.remove_at_position(0), "two")

    @spectest(1)
    @timeout(0.1)
    def test_efficiency_add_last(self):
        huge_linked_list.add_last("test")
        self.assertEqual(huge_linked_list.get_tail(), "test")

    @spectest(1)
    @timeout(0.1)
    def test_efficiency_remove_last(self):
        self.assertNotEqual(huge_linked_list.remove_last(), None)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
