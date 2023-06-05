class LinkedList:

    def remove_last(self):
        if self.size == 1:
            self.head = None
            self.tail = self.head
        else:
            cur = self.head
            while cur.next != self.tail:
                cur = cur.next
            cur.next = None
            self.tail = cur
        self.size -= 1
