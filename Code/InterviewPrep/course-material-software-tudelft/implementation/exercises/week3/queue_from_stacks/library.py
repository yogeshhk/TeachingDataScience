class Stack:

    def __init__(self):
        self.__items = []

    def push(self, item):
        self.__items.append(item)

    def pop(self):
        return self.__items.pop()

    def top(self):
        return self.__items[-1]

    def __len__(self):
        return len(self.__items)

    def is_empty(self):
        return self.__items == []
