def PQ:

    def __init__(self):
        self.li = list()

    def __len__(self):
        return len(self.li)

    def add(self, item):
        self.li.append(item)

    def min(self):
        return min(self.li)

    def remove_min(self):
        result = min(self.li)
        self.li.remove(result)
        return result
