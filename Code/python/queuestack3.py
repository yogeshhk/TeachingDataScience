class StackQueue:

    def __init__(self):
        self.s1 = Stack()
        self.s2 = Stack()

    def enqueue(self, item):
        self.s1.push(item)

    def dequeue(self):
        if len(self.s2) == 0:
            while len(self.s1) > 0:
                self.s2.push(self.s1.pop())
        return self.s2.pop()
