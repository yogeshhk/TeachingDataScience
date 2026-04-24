class LinkedList:

    def add_first(self, val: int):
        if self.head is None:
            self.head = Node(val, None)
            self.tail = self.head
        else:
            self.head = Node(val, self.head)
        self.size += 1
