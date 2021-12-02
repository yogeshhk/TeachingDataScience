from collections import deque


class Stack:

    def __init__(self):
        self.__items = []

    def is_empty(self):
        return self.__items == []

    def push(self, item):
        self.__items.append(item)

    def pop(self):
        return self.__items.pop()

    def top(self):
        return self.__items[-1]

    def __len__(self):
        return len(self.__items)


class SpecialQueue:

    def __init__(self):
        self.__items = deque()

    def is_empty(self):
        return len(self.__items) == 0

    def enqueue_front(self, item):
        self.__items.append(item)

    def enqueue_back(self, item):
        self.__items.appendleft(item)

    def dequeue(self):
        return self.__items.pop()

    def first(self):
        return self.__items[-1]

    def __len__(self):
        return len(self.__items)
