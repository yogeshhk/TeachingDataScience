from __future__ import annotations
from typing import List, Set, Dict
from operator import attrgetter


class Node:

    val: int
    neighbours: Set[Node]

    def __init__(self, val: int):
        self.val = val
        self.neighbours = set()

    def add_neighbour(self, node: Node):
        self.neighbours.add(node)

    def get_neighbours(self):
        return sorted(self.neighbours, key=attrgetter("val"))


class Graph:

    nodes: Dict[int, Node]

    def __init__(self):
        self.nodes = {}

    def add_node(self, val: int):
        self.nodes[val] = Node(val)

    def get_node(self, val: int):
        return self.nodes[val]

    def add_edge(self, id1: int, id2: int):
        self.nodes[id1].add_neighbour(self.nodes[id2])
        self.nodes[id2].add_neighbour(self.nodes[id1])

    def all_nodes(self) -> List[Node]:
        return sorted(self.nodes.values(), key=attrgetter("val"))
