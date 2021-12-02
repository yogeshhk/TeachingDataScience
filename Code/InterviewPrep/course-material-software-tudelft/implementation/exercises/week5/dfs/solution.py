from collections import deque
from typing import List
from decorators import empty
from .library import Graph, Node


@empty
def depth_first(g: Graph, n: Node) -> List[Node]:
    s = deque()
    s.append(n)
    seen = set()
    res = []
    while not len(s) == 0:
        cur = s.pop()
        if cur not in seen:
            seen.add(cur)
            res.append(cur.val)
            for node in reversed(g.get_neighbours(cur)):
                s.append(node)
    return res
