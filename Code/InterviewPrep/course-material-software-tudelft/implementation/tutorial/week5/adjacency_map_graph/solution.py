from __future__ import annotations
from typing import Dict, Set

from decorators import empty


class Vertex:

    adj: Dict[int, Vertex]
    idx: int

    def __init__(self, idx: int):
        self.idx = idx
        self.adj = {}

    def get_neighbours(self):
        return self.adj


class Graph:

    # Amount of vertices in this graph
    size: int
    # Vertices in this graph
    vertices: Dict[int, Vertex]

    @empty
    # Constructor of the graph
    def __init__(self):
        self.vertices = {}
        self.size = 0

    @empty
    # Adds a new vertex to the graph,
    # This vertex will use the next available number as index in the dictionary
    def add_vertex(self):
        v = Vertex(self.size)
        self.vertices[self.size] = v
        self.size += 1

    @empty
    # Adds a new edge to the graph,
    # v1 and v2 are the indices of the vertices
    def add_edge(self, v1: int, v2: int):
        if v1 == v2:
            return
        self.vertices[v1].adj[v2] = self.vertices[v2]
        self.vertices[v2].adj[v1] = self.vertices[v1]

    @empty
    # Remove an existing edge from the graph,
    # v1 and v2 are the indices of the vertices
    def remove_edge(self, v1: int, v2: int):
        self.vertices[v1].adj.pop(v2)
        self.vertices[v2].adj.pop(v1)

    @empty
    # Checks whether an edge exists in the graph,
    # v1 and v2 are the indices of the vertices
    def contains_edge(self, v1: int, v2: int):
        return v2 in self.vertices[v1].adj
