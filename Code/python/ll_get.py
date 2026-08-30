class LinkedList:

    def get(self, index: int) -> int:
        if index >= self.size:
            return None  # Invalid index
        cur = self.head
        cnt = 0
        while cnt < index:
            cur = cur.next
            cnt += 1
        return cur.value
