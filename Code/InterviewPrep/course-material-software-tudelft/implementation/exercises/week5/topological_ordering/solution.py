from typing import List

from decorators import empty

from .library import Graph


@empty
# Returns a list with the values of the vertices in the graph,
# The order in this list will be a topological ordering of this graph.
def topological_ordering(g: Graph) -> List[int]:
    # Keep a list of the degree of incoming edges for each node
    in_degree = [len(x.incoming) for x in g.all_nodes()]
    # Queue to store all nodes that we are currently processing
    queue = []
    # Check for all nodes, add them to the queue if they have an in degree of 0
    for idx, val in enumerate(in_degree):
        if val == 0:
            queue.append(idx)

    # List to store result
    res = []

    # While we still have nodes to process
    while queue:
        # Get the first node
        u = queue.pop()
        # Add it to the result
        res.append(u)

        # Decrease the in degree of all nodes that had an incoming edge from the current node
        for x in g.get_node(u).outgoing:
            in_degree[x.val] -= 1
            # If the in degree becomes 0, add the node to the queue
            if in_degree[x.val] == 0:
                queue.append(x.val)
    return res
