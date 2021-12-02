from collections import deque
from typing import List
from decorators import empty
from .library import Graph, Node


@empty
def breadth_first(g: Graph, n: Node) -> List[Node]:
    q = deque()
    q.append(n)
    seen = {n}
    res = []
    while not len(q) == 0:
        cur = q.popleft()
        res.append(cur.val)
        for node in g.get_neighbours(cur):
            if node not in seen:
                q.append(node)
                seen.add(node)
    return res
