#### 1) Corresponding run time equation for function and explanation of all terms

The run time of this function depends on how many neighbours each vertex has.
Let \\(\deg(v)\\) denote the degree of \\(v\\) (= amount of neighbours).
Let \\(n\\) be the amount of vertices and \\(m\\) be the amount of edges in `graph`.
Let \\(C\\) be the amount of connected components in `graph`.
Let \\(C_i\\) be the set of vertices in the \\(i^\text{th}\\) connected component.
Let \\(V\\) be the set of all vertices in `graph`.

- Lines 4, 7, 8, 9, 19 - \\(c_0\\) - Initializing call, constant-time assignment of variables, initializing loop, returning from function
- Line 6 - \\(c_1 n\\) - Set creation with \\(n\\) elements
- Line 9 - This loop is ran as many times as there are connected components (\\(C\\) times, for every component \\(C_i\\))
    - Line 9, 10, 11, 12 - \\(c_2\\) - checking loop condition, assignment of variables, popping from set, adding to set, initializing loop
    - Line 12 - This loop is ran as many times as there are vertices in the current connected component (\\(|C_i|\\) times, for every \\(v \in C_i\\))
        - Lines 12, 13, 14, 15 - \\(c_3\\) - checking loop condition, assignment of variables, popping from set, indexing dictionary, initializing loop
        - Line 15 - This loop is ran as many times as there are neighbours of the current vertex (\\(\deg(v)\\) times)
            - Lines 15, 16, 17, 18 - \\(c_4\\) checking loop condition, evaluating `if`-condition (checking set containment), possibly removing an item from a set and adding an item to an other set, all constant time

We first state the run time equation in terms of the above observations, and will then work out the sum terms.

$$
\begin{align}
T(n) &= c_0 + c_1n + \sum_{i=1}^C\left[ c_2 + \sum_{v \in C_i} \left( c_3 + c_4\deg(v) \right) \right] \\\\
T(n) &= c_0 + c_1n + c_2C + \sum_{i=1}^C \sum_{v \in C_i} \left( c_3 + c_4\deg(v) \right) \\\\
     & \text{all vertices are distinct connected components, so we can merge the sum terms} \\\\
T(n) &= c_0 + c_1n + c_2C + \sum_{v \in V} \left( c_3 + c_4\deg(v) \right) \\\\
T(n) &= c_0 + c_1n + c_2C + c_3n + \sum_{v \in V} c_4\deg(v) \\\\
     & \text{use property of undirected graphs:} \sum_{v \in V} \deg(v) = 2m \\\\
T(n) &= c_0 + c_1n + c_2C + c_3n + 2c_4m \\\\
\end{align}
$$

#### 2) Run time complexity in terms of Big Oh notation

After simplification, the constant terms and factors can be disregarded as n grows to infinity. This leaves us with \\(T(n) = C + n + m\\).
The term \\(C\\) is upper bounded by \\(n\\), as there can never be more connected components than vertices, so it can be dropped as well.
Thus, the run time complexity is \\(\mathcal O(n + m)\\).
