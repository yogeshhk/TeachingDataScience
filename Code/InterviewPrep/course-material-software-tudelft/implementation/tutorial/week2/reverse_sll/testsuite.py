import unittest

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import SLL


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_one_element(self):
        sll = SLL()
        sll.add_first("42")
        self.assertEqual(sll.get_head(), "42")
        self.assertEqual(sll.head.next_node, None)
        sll.reverse()
        self.assertEqual(sll.get_head(), "42")
        self.assertEqual(sll.head.next_node, None)

    @yourtest
    @spectest(1)
    def test_reverse_three_elements(self):
        sll = SLL()
        sll.add_first("three")
        sll.add_first("two")
        sll.add_first("one")
        self.assertEqual(sll.get_head(), "one")
        self.assertEqual(sll.head.next_node.value, "two")
        self.assertEqual(sll.head.next_node.next_node.value, "three")
        sll.reverse()
        self.assertEqual(sll.get_head(), "three")
        self.assertEqual(sll.head.next_node.value, "two")
        self.assertEqual(sll.head.next_node.next_node.value, "one")

    @yourtest
    @spectest(1)
    def test_double_reverse(self):
        original_sll = SLL()
        reversed_sll = SLL()

        original_sll.add_first("Julius Caesar")
        reversed_sll.add_first("Julius Caesar")
        original_sll.add_first("Alexander the Great")
        reversed_sll.add_first("Alexander the Great")
        original_sll.add_first("Genghis Khan")
        reversed_sll.add_first("Genghis Khan")
        original_sll.add_first("Oda Nobunaga")
        reversed_sll.add_first("Oda Nobunaga")
        original_sll.add_first("Napoleon Bonaparte")
        reversed_sll.add_first("Napoleon Bonaparte")
        original_sll.add_first("Attila the Hun")
        reversed_sll.add_first("Attila the Hun")

        reversed_sll.reverse()
        reversed_sll.reverse()

        temp_original = original_sll.head
        temp_reversed = reversed_sll.head
        while temp_original is not None and temp_reversed is not None:
            self.assertEqual(temp_original.value, temp_reversed.value)
            temp_original = temp_original.next_node
            temp_reversed = temp_reversed.next_node


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
