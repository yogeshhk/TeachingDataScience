import heapq as hq


class Task:
    def __init__(self, d: int):
        self.duration = d


class PriorityQueue:
    def __init__(self):
        self.q = []

    def enqueue(self, item):
        hq.heappush(self.q, item)

    def dequeue(self):
        return hq.heappop(self.q)

    def first(self):
        return self.q[0]

    def __len__(self):
        return len(self.q)

    def is_empty(self):
        return len(self.q) == 0
