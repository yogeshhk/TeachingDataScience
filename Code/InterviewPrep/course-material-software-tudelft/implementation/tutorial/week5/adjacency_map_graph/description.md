In this exercise you have to implement a graph structure using an adjacency map.

The Graph has a few attributes that you need to use:

- `vertices`, this is a map of vertices, the key is in the index of the vertex, and the value is the vertex.
- `size`, this is the current amount of vertices in the graph.

The Vertex has a few attributes that you need to use:

- `idx`, this is the index of the vertex.
- `adj`, this is the adjacency map, the key is the index of a neighbouring vertex, and the value is the vertex.

You will need to implement the following functions of the graph:

- `__init__`, this is the constructor of the graph. You will have to initialize the attributes that this graph has.
- `add_vertex`, this function adds a vertex to the graph. The index number of the new vertex will be one more than the highest index of all existing vertices.
- `add_edge`, this function adds an edge to graph. Note that a vertex can't share an edge with itself.
- `remove_edge`, this function removes an edge from the graph.
- `contain_edge`, this function checks if there is an edge between two vertices.
