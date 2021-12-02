class Peg:

    def __init__(self):
        self.__items = []

    def is_empty(self):
        return self.__items == []

    def push(self, item):
        if not self.is_empty():
            assert self.__items[-1] >= item
        self.__items.append(item)

    def pop(self):
        return self.__items.pop()

    def size(self):
        return len(self.__items)
