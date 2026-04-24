class linkedList:

    def contains(self, val: int) -> bool:
        cur = self.head
        while cur is not None:
            if cur.value == val:
                return True
            cur = cur.next
        return False
