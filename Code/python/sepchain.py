class HashMap:

    def get_from_bucket(self, j, k):
        bucket = self.table[j]
        for node in bucket:
            if node.key == k:
                return node.value
        return None

    def put_in_bucket(self, j, k, v):
        if self.table[j] is None:
            self.table[j] = LinkedList()
        bucket = self.table[j]
        for node in bucket:
            if node.key == k:
                node.value = v  # It was an update
                return
        bucket.add(Node(k, v))
