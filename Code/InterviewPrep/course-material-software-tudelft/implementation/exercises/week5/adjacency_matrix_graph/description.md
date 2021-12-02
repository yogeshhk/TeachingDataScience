In this exercise you have to implement a graph structure using an adjacency matrix.

The Graph has a few variables that you need to use:

- `adj`, this is the adjacency matrix, using booleans to show whether there is an edge or not. There is an edge between vertex v_i and v_j, if and only if `adj[i][j]` and `adj[j][i]` are both `True`.
- `size`, this is the current amount of vertices in the graph.

You will need to implement the following functions of the graph:

- `__init__`, this is the constructor of the graph. This takes as input an integer, representing the number of vertices in this graph, given this number you will have to create an adjacency matrix with the correct dimensions for this graph.
- `add_vertex`, this function adds a vertex to the graph, the index number of the vertices are auto incrementing, which means that the index of the new vertex will be the highest index of all existing vertices + 1.
- `add_edge`, this function adds an edge to graph. 
- `remove_edge`, this function removes an edge from the graph.
- `contains_edge`, this function checks if there is an edge between two vertices.