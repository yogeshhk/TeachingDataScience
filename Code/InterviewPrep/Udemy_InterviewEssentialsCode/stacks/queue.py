class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class Queue: 
    def __init__(self): 
        self.head = None
        self.tail = None
      
    def push(self, value): 
        new_node = Node(value)
        if self.tail == None: 
            self.head = new_node 
            self.tail = self.head
        else: 
            self.tail.next = new_node
            self.tail = new_node
      
    def pop(self): 
        if self.head == None: 
            return None
        else: 
            popped = self.head 
            self.head = self.head.next
            if (self.head == None):
                self.tail = None
            return popped.val
      
    def peek(self): 
        if self.head == None: 
            return None
        else: 
            return self.head.val 
     
queue = Queue()
queue.push(1)
queue.push(2)
queue.push(3)
print(queue.peek())
queue.pop()
print(queue.peek())
queue.push(4)
queue.pop()
print(queue.peek())