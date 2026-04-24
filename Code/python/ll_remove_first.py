class LinkedList:

    def remove_first(self):
        if self.size == 1:
            self.head = None
            self.tail = self.head
        else:
            self.head = self.head.next
        self.size -= 1
