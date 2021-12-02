from __future__ import annotations

from typing import List, Set


class Node:
    def __init__(self, val: int):
        self.val = val
        # Outgoing edges
        self.outgoing = set()
        # Incoming edges
        self.incoming = set()

    def add_outgoing(self, node: Node):
        self.outgoing.add(node)

    def add_incoming(self, node: Node):
        self.incoming.add(node)


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = set()

    # Index of nodes is auto incrementing
    def add_node(self):
        val = len(self.nodes)
        self.nodes[val] = Node(val)

    def get_node(self, val: int) -> Node:
        return self.nodes[val]

    # Adds a directed edge from the vertex with id1 to the vertex with id2
    def add_edge(self, id1: int, id2: int):
        self.nodes[id1].add_outgoing(self.nodes[id2])
        self.nodes[id2].add_incoming(self.nodes[id1])
        self.edges.add((id1, id2))

    def all_nodes(self) -> List[Node]:
        return sorted(self.nodes.values(), key=lambda node: node.val)

    def all_edges(self) -> Set[(int, int)]:
        return self.edges

    def get_neighbours(self, node: Node) -> List[Node]:
        return sorted(self.nodes[node.val].neighbours)
