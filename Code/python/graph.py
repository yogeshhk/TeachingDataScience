class Graph:

    def __init__(self):
        self.vertices = dict()

    def addVertex(self, v):
        self.vertices[v.name] = v

    def vertices(self):
        return list(self.vertices.values())
