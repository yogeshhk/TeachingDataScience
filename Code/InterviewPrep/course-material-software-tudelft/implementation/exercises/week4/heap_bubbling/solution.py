from typing import List

from decorators import empty


@empty
# Restores the heap property in a heap represented as a List.
# The parameter `root` indicates the index that represents the position at which the heap might be invalid.
def down_heap(heap: List[int], root: int = 0) -> None:
    # Using a handy property of binary trees:
    # If we have a node at index i, its children will be at indices 2 * i + 1 and 2 * i + 2.
    left = 2 * root + 1
    right = 2 * root + 2

    largest = root
    if left < len(heap) and heap[left] > heap[largest]:
        largest = left
    if right < len(heap) and heap[right] > heap[largest]:
        largest = right

    # Do a recursive call if the heap property is invalid at root
    if largest != root:
        heap[root], heap[largest] = heap[largest], heap[root]
        down_heap(heap, largest)
