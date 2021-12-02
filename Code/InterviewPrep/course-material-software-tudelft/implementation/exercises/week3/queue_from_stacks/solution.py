from decorators import empty, remove
from .library import Stack


class Queue:

    def __init__(self):
        self.s1 = Stack()
        self.s2 = Stack()

    @empty
    # Puts the given item in the queue
    def enqueue(self, item):
        # We only put items in s1. s2 is used for output only, as explained in the comments of the flip function.
        self.s1.push(item)

    @empty
    # Removes the first item from the queue
    def dequeue(self):
        # We only return items from s2. If this is empty, we collect the items that are in s1.
        if self.s2.is_empty():
            self.flip()
        # Now we can return an item
        return self.s2.pop()

    @empty
    # Returns the first element in the queue without removing it
    def first(self):
        # We only return items from s2. If this is empty, we should collect the items that are in s1.
        if self.s2.is_empty():
            self.flip()
        # Now we can return an item
        return self.s2.top()

    @remove
    # Empties s1 and puts all items in s2, thereby effectively making s2 the reverse of the original s1.
    # This maintains the order we need in a queue: the item that was entered in s1 first ends on top of s2,
    # from which we return items. The last item entered is returned first.
    # Try drawing this process on paper to see why it works.
    def flip(self):
        while not self.s1.is_empty():
            self.s2.push(self.s1.pop())

    @empty
    # Returns the amount of elements in the queue
    def __len__(self):
        return len(self.s1) + len(self.s2)

    @empty
    # Returns true if the queue is empty
    def is_empty(self):
        return len(self) == 0
