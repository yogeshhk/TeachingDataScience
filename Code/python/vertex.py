class Vertex:

    def __init__(self, name):
        self.name = name
        self.neighbours = dict()

    def addEdge(self, to, weight):
        self.neighbours[to] = weight

    def neighbours(self):
        return list(self.neighbours.keys())

    def edges(self):
        for (to, weight) in self.neighbours.items():
            yield (to, weight)

    def isNeighbour(self, v):
        return v in self.neighbours

    def getWeight(self, v):
        if self.isNeighbour(v):
            return self.neighbours[v]
        return None  # Or whatever else is appropriate


v = Vertex(1)

v.addEdge(2, 3)
v.addEdge(3, 5)

print(list(v.edgesIterator()))
