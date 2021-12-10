class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class Stack: 
    def __init__(self): 
        self.head = None
      
    def push(self, value): 
        if self.head == None: 
            self.head = Node(value) 
        else: 
            new_node = Node(value) 
            new_node.next = self.head 
            self.head = new_node 
      
    def pop(self): 
        if self.head == None: 
            return None
        else: 
            popped = self.head 
            self.head = self.head.next
            return popped.val
      
    def peek(self): 
        if self.head == None: 
            return None
        else: 
            return self.head.val 
      
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.peek())
stack.pop()
print(stack.peek())
stack.push(4)
print(stack.peek())