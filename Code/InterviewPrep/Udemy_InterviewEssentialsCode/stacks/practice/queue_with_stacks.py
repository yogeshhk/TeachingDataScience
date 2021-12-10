class Queue():
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        while (self.stack2):
            self.stack1.append(self.stack2.pop())
        self.stack1.append(x)

    def pop(self):
        while (self.stack1):
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self):
        while (self.stack1):
            self.stack2.append(self.stack1.pop())
        return self.stack2[-1]
        
    def empty(self):
        return self.stack1 == [] and self.stack2 == []

queue = Queue()
queue.push(1)
queue.push(2)
print(queue.peek())
print(queue.pop())
print(queue.empty())
