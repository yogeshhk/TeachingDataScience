def PQ:
    def __init__(self):
        self.li = list()

    def __len__(self):
        return len(self.li)

    def add(self, item):
        i = 0
        while self.li[i] < item:
            i += 1
        self.li.insert(i, item)

    def min(self):
        return self.li[0]

    def remove_min(self):
        return self.li.pop(0)
