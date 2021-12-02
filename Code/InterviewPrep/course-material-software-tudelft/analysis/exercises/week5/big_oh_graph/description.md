For the code snippet below, work out what the worst-case run time complexity is in terms of big-Oh notation.

```python
from typing import Dict, List


def connected_components(
        graph: Dict[int, List[int]]) -> int:
  nodes = set(graph.keys())
  queue = set()
  res = 0
  while len(nodes) > 0:
    res += 1
    queue.add(nodes.pop())
    while len(queue) > 0:
      node = queue.pop()
      neighbours = graph[node]
      for n in neighbours:
        if n in nodes:
          nodes.remove(n)
          queue.add(n)
  return res
```

The `graph` in this function is represented using a single adjacency map and is undirected.
An example `graph` could be `{0: [1], 1: [0], 2: []}`, where nodes `0` and `1` are connected to each other and node `2` is isolated.

1) Derive the run time equation of this code and explain all the terms.

2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.
