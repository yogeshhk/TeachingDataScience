class HashMap:

    def __init__(self):
        self.size = 0
        self.table = [None] * 11

    def __len__(self):
        return self.size

    def hashfunc(self, k):
        return hash(k) % len(self.table)

    def get(self, k):
        j = self.hashfunc(k)
        return self.get_from_bucket(j, k)

    def put(self, k, v):
        j = self.hashfunc(k)
        self.put_in_bucket(j, k, v)
        if self.size > len(self.table) // 2:
            self.resize(2 * len(self.table))
