#   Created by Elshad Karimov on 02/06/2020.
#   Copyright © 2020 AppMillers. All rights reserved.

# Use a single list to implement three stacks.

class MultiStack:
    def __init__(self, stacksize):
        self.numberstacks = 3
        self.custList = [0] * (stacksize * self.numberstacks)
        self.sizes = [0] * self.numberstacks
        self.stacksize = stacksize
    
    def isFull(self, stacknum):
        if self.sizes[stacknum] == self.stacksize:
            return True
        else:
            return False
    
    def isEmpty(self, stacknum):
        if self.sizes[stacknum] == 0:
            return True
        else:
            return False
    
    def indexOfTop(self, stacknum):
        offset = stacknum * self.stacksize
        return offset + self.sizes[stacknum]- 1
    
    def push(self, item, stacknum):
        if self.isFull(stacknum):
            return "The stack is full"
        else:
            self.sizes[stacknum] += 1
            self.custList[self.indexOfTop(stacknum)] = item
    
    def pop(self, stacknum):
        if self.isEmpty(stacknum):
            return "The stack is empty"
        else:
            value = self.custList[self.indexOfTop(stacknum)]
            self.custList[self.indexOfTop(stacknum)] = 0
            self.sizes[stacknum] -= 1
            return value
    
    def peek(self, stacknum):
        if self.isEmpty(stacknum):
            return "The stack is empty"
        else:
            value = self.custList[self.indexOfTop(stacknum)]
            return value


customStack = MultiStack(6)
print(customStack.isFull(0))
print(customStack.isEmpty(1))
customStack.push(1, 0)
customStack.push(2, 0)
customStack.push(3, 2)
print(customStack.pop(0))
        
