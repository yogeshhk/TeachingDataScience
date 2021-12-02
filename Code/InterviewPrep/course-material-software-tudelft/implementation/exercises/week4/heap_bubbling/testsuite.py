import random
import unittest
from typing import List

from decorators import remove, spectest, timeout, yourtest
from weblabTestRunner import TestRunner
from .solution import down_heap


@remove
def is_valid_heap(heap: List[int]) -> bool:
    i = 0
    while True:
        if 2 * i + 2 >= len(heap):
            return True
        if 2 * i + 1 >= len(heap):
            return heap[i] > heap[2 * i + 1]
        if heap[i] < heap[2 * i + 1] or heap[i] < heap[2 * i + 2]:
            return False
        i += 1


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    # This test represents the following heap:
    #      2
    #     / \
    #    1   3
    # The heap property is invalid at the root and should be restored.
    # The resulting heap is:
    #      3
    #     / \
    #    1   2
    def test_small(self):
        heap = [2, 1, 3]
        res = [3, 1, 2]
        down_heap(heap)
        self.assertEqual(res, heap)

    @yourtest
    @spectest(1)
    # This test represents the following heap:
    #       49
    #      /  \
    #    12    38
    #   /
    # 24
    # The heap property is invalid at the node with index 1 and should be restored.
    # The resulting heap will have the elements `12` and `24` swapped.
    def test_other_root(self):
        heap = [49, 12, 38, 24]
        res = [49, 24, 38, 12]
        down_heap(heap, 1)
        self.assertEqual(res, heap)

    @spectest(1)
    def test_one(self):
        heap = [42]
        res = [42]
        down_heap(heap)
        self.assertEqual(res, heap)

    @spectest(1)
    def test_valid_heap(self):
        heap = [5, 3, 1, 2]
        res = heap.copy()
        down_heap(heap)
        self.assertEqual(res, heap)

    @spectest(1)
    def test_multiple_levels(self):
        heap = [2, 6, 7, 4, 3, 5, 1]
        res = [7, 6, 5, 4, 3, 2, 1]
        down_heap(heap)
        self.assertEqual(res, heap)

    @spectest(1)
    def test_equal_numbers(self):
        heap = [42 for _ in range(42)]
        res = heap.copy()
        down_heap(heap)
        self.assertEqual(res, heap)

    @spectest(3)
    def test_multiple_bubbling(self):
        heap = list(range(43))
        heap.reverse()
        for i in range(42, 0, -1):
            heap[0], heap[i] = heap[i], heap[0]
            heap.pop()
            down_heap(heap)
            self.assertTrue(is_valid_heap(heap), f"Should be a valid heap with length {i}")

    @timeout(1)
    @spectest(5)
    def test_random_heap(self):
        random.seed(193462)
        for i in range(15):
            size = 2 ** i
            heap = [random.randint(0, size * 2) for _ in range(size)]
            expected = sorted(heap.copy(), reverse=True)
            for root in range(size // 2, -1, -1):
                down_heap(heap, root)
            self.assertTrue(is_valid_heap(heap), f"Should be a valid heap with size {size}")
            res = []
            for j in range(size - 1, 0, -1):
                res.append(heap[0])
                heap[0], heap[j] = heap[j], heap[0]
                heap.pop()
                down_heap(heap)
            res.append(heap[0])
            self.assertEqual(expected, res, f"List should be reversely sorted with size {size}")


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
