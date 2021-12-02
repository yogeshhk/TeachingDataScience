from collections import deque


class Queue:
    def __init__(self):
        self.q = deque()

    def enqueue(self, item):
        self.q.appendleft(item)

    def dequeue(self):
        return self.q.pop()

    def first(self):
        return self.q[-1]

    def __len__(self):
        return len(self.q)

    def is_empty(self):
        return len(self.q) == 0
