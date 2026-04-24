class LinkedList:

    def add_last(self, val: int):
        if self.head is None:
            self.head = Node(val, None)
            self.tail = self.head
        else:
            self.tail.next = Node(val, None)
            self.tail = self.tail.next
        self.size += 1
