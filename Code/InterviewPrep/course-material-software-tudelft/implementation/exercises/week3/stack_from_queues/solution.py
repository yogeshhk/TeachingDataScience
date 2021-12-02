from decorators import empty, remove
from .library import Queue


class Stack:

    def __init__(self):
        self.q1 = Queue()
        self.q2 = Queue()

    @empty
    # Puts the given item on the stack
    def push(self, item):
        # We only put items in q1. q2 is used for output only, as explained in the comments of the flip function.
        self.q1.enqueue(item)

    @empty
    # Removes the first item from the stack
    def pop(self):
        # We only return items from q2. If this is empty, we collect the items that are in q1.
        if self.q2.is_empty():
            self.flip()
        # Now we can return an item
        return self.q2.dequeue()

    @empty
    # Returns the first element on the stack without removing it
    def top(self):
        # We only return items from q2. If this is empty, we should collect the items that are in q1.
        if self.q2.is_empty():
            self.flip()
        # Now we can return an item
        return self.q2.first()

    @remove
    # Recursively empties q1 and puts all items in q2. Because of the recursive call between dequeue and enqueue,
    # the order is reversed. The reverse of q1 is the order in which we want to return items:
    # the item last entered in q1 is now first in q2 (enqueued as last).
    # Try drawing this process on paper to see why it works.
    def flip(self):
        if not self.q1.is_empty():
            temp = self.q1.dequeue()
            self.flip()
            self.q2.enqueue(temp)

    @empty
    # Returns the amount of elements in the stack
    def __len__(self):
        return len(self.q1) + len(self.q2)

    @empty
    # Returns true if the queue is empty
    def is_empty(self):
        return len(self) == 0
