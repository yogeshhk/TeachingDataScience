The social media network ADS (All Decent Students) stores all connections between users in a graph. The nodes represent people and there's an edge between two people if they are connected on the platform.

Implement the function `connections` that returns the IDs (stored as value of the node) of the `k` closest connections you have (excluding yourself). People directly connected to you are closest, people connected to connections of yours are next, then people who are connected to the previous group and so on.

In the library you will find implementations for the `Graph` and `Node` classes. Each node stores its neighbours.

If you have multiple choices of which node to pick, you should choose them in ascending order of value. The function `get_neighbours` in the library class `Graph` already returns the neighbours in ascending order of their values.

Hint: think about the graph traversal algorithms treated in the lectures.
