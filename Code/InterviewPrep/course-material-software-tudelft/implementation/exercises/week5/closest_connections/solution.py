from collections import deque
from typing import Set
from decorators import empty
from .library import Graph, Node


@empty
# Return the values of the k closest nodes in the graph starting from (but excluding) your_node
def connections(g: Graph, k: int, your_node: Node) -> Set[int]:
    q = deque([your_node])
    seen = {your_node}
    res = set()
    count = 0
    while not len(q) == 0 and count <= k:
        cur = q.popleft()
        res.add(cur.val)
        for node in cur.get_neighbours():
            if node not in seen:
                q.append(node)
                seen.add(node)
        count += 1
    res.remove(your_node.val)
    return res
