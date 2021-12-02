import random
import unittest
from typing import List

from decorators import spectest, solution_only, timeout, yourtest
from weblabTestRunner import TestRunner
from .library import SLL
from .solution import sort


def add_list(ll: SLL, el: List[int]):
    for i in range(len(el) - 1, -1, -1):
        ll.add_first(el[i])


def get_list(ll: SLL) -> List[int]:
    result = []
    c = ll.head
    while c is not None:
        result.append(c.value)
        c = c.next_node
    return result


if solution_only:
    random.seed(42)
    huge_linked_list = SLL()
    huge_list = list(random.randint(0, 10000) for _ in range(1000))
    add_list(huge_linked_list, huge_list)
    huge_list.sort()


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        ll = SLL()
        result = [2, 3, 1]
        add_list(ll, result)
        sort(ll)
        result.sort()
        self.assertEqual(result, get_list(ll))

    @yourtest
    @spectest(1)
    def test_empty(self):
        ll = SLL()
        result = []
        sort(ll)
        self.assertEqual(result, get_list(ll))

    @yourtest
    @spectest(1)
    def test_one_element(self):
        ll = SLL()
        result = [42]
        add_list(ll, result)
        sort(ll)
        self.assertEqual(result, get_list(ll))

    @spectest(1)
    def test_sorted(self):
        ll = SLL()
        result = [24, 42, 242, 424, 1337, 1337, 1733, 3137, 3173, 3317, 3371]
        add_list(ll, result)
        sort(ll)
        self.assertEqual(result, get_list(ll))

    @spectest(1)
    def test_reversed(self):
        ll = SLL()
        result = [5, 4, 3, 2, 1]
        add_list(ll, result)
        sort(ll)
        result.sort()
        self.assertEqual(result, get_list(ll))

    @spectest(5)
    @timeout(1)
    def test_large(self):
        sort(huge_linked_list)
        self.assertEqual(huge_list, get_list(huge_linked_list))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
