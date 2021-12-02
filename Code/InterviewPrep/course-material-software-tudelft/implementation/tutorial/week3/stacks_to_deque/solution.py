from decorators import empty, remove
from .library import Stack


class Deque:

    def __init__(self):
        self.s1 = Stack()
        self.s2 = Stack()

    @empty
    # Add an item to the left of the queue
    def add_first(self, item):
        self.s1.push(item)

    @empty
    # Add an item to the right of the queue
    def add_last(self, item):
        self.s2.push(item)

    @empty
    # Remove an item from the left of the queue
    def remove_first(self):
        if self.s1.is_empty():
            self.fill_s1()
        return self.s1.pop()

    @empty
    # Remove an item from the right of the queue
    def remove_last(self):
        if self.s2.is_empty():
            self.fill_s2()
        return self.s2.pop()

    @empty
    # Returns the first element without removing it
    def first(self):
        if self.s2.is_empty():
            self.fill_s2()
        return self.s2.top()

    @empty
    # Returns the last element without removing it
    def last(self):
        if self.s1.is_empty():
            self.fill_s1()
        return self.s1.top()

    @remove
    # Fills s1 from s2
    def fill_s1(self):
        while not self.s2.is_empty():
            self.s1.push(self.s2.pop())

    @remove
    # Fills s2 from s1
    def fill_s2(self):
        while not self.s1.is_empty():
            self.s2.push(self.s1.pop())

    @empty
    # Return the amount if items in the deque
    def __len__(self):
        return len(self.s1) + len(self.s2)
