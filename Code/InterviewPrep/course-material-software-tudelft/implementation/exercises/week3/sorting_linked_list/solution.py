from decorators import empty
from .library import SLL


@empty
# Sorts the linked list in ascending order using bubble sort
def sort(xs: SLL):
    for i in range(xs.size, 0, -1):
        c = xs.head
        n = xs.head.next_node
        for j in range(i - 1):
            if n.value < c.value:
                c.value, n.value = n.value, c.value
            c = n
            n = n.next_node
