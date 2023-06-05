class HashMap:

    def get_from_bucket(self, j, k):
        while self.table[j] is not None:
            if node.key == k:
                return node.value
            j = (j + 1) % len(self.table)
        return None

    def put_in_bucket(self, j, k, v):
        while self.table[j] is not None:
            if self.table[j].key == k:
                self.table[j].value = v  # It was an update
                return
            j = (j + 1) % len(self.table)
        self.table[j] = Node(k, v)
