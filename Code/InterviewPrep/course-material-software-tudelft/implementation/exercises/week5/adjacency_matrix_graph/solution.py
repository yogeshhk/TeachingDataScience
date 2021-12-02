from typing import List

from decorators import empty


class Graph:

    # Adjacency matrix
    adj: List[List[bool]]
    # Amount of vertices in the graph
    size: int

    @empty
    # Constructor of the graph
    def __init__(self, size: int):
        self.adj = [[False for _ in range(size)] for _ in range(size)]
        self.size = size

    @empty
    # Add a new vertex to the graph,
    # The vertex will use the largest index of the existing vertices as index
    def add_vertex(self):
        for x in self.adj:
            x.append(False)
        self.size += 1
        self.adj.append([False for _ in range(self.size)])

    @empty
    # Add a new edge to the graph,
    # v1 and v2 are the indices of the vertices
    def add_edge(self, v1: int, v2: int):
        self.adj[v1][v2] = True
        self.adj[v2][v1] = True

    @empty
    # Remove an existing edge from the graph,
    # v1 and v2 are the indices of the vertices
    def remove_edge(self, v1: int, v2: int):
        self.adj[v1][v2] = False
        self.adj[v2][v1] = False

    @empty
    # Checks whether an edge exists in the graph,
    # v1 and v2 are the indices of the vertices
    def contains_edge(self, v1: int, v2: int) -> bool:
        return self.adj[v1][v2]
