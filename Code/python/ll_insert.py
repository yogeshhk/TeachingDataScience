class LinkedList:

    def insert(self, index: int, value: int):
        if index >= self.size:
            self.append(value)
        elif index == 0:
            self.add_first(value)
        else:
            cur = self.head
            cnt = 0
            while cnt < index-1:
                cur = cur.next
                cnt += 1
            newNode = Node(value, cur.next)
            cur.next = newNode
            self.size += 1
