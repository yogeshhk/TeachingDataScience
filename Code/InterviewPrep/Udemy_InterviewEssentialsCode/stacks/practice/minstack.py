class MinStack():
    def __init__(self):
        self.items = []
        self.mins = []

    def push(self, x):
        self.items.append(x)
        if len(self.mins) == 0:
            self.mins.append(x)
        else:
            self.mins.append(min(x, self.getMin()))

    def pop(self):
        self.items.pop()
        self.mins.pop()

    def top(self):
        return self.items[-1]

    def getMin(self):
        return self.mins[-1]

minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin())
minStack.pop()
print(minStack.top())
print(minStack.getMin())