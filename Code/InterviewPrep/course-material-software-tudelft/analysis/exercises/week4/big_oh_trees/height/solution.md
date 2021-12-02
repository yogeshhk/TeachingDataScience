1) Derive the run time equation of this code and explain all the terms.

We take \\(n\\) to be the number of nodes in the tree.

The algorithm determines the height of the tree. The base case is an empty tree, so \\(n = 0\\).

Since this tree is not necessarily balanced, we cannot assume a recursive call to one of the child nodes is \\(T(n/2)\\) work. In the worst case the tree is linear: all nodes have only 1 child node and the height of the tree is equal to the amount of nodes.

\\(T(n)=\begin{cases}
    c_0 & \text{if $n=0$}.\\\\
    c_1 + T(n-1) & \text{otherwise}.
  \end{cases}\\)
  
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​

\\(T(n) = c_1 + T(n-1)\\)

We substitute \\(T(n-1)\\):

\\(T(n) = c_1 + c_1 + T(n-2)\\)

We substitute \\(T(n-2)\\):

\\(T(n) = 3 c_1 + + T(n-3)\\)

For \\(k\\) repetitions of substitution:

\\(T(n) = k c_1 + + T(n-k)\\)

Take \\(k = n\\) to arrive at the base case:

\\(T(n) = n c_1 + + T(n-n)\\)

\\(T(n) = n c_1 + + T(0)\\)

\\(T(n) = n c_1 + + c_0\\)

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n\\), which is linear growth. Thus, the run time complexity is \\(\mathcal{O}(n)\\).