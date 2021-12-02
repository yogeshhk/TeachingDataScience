1) Derive the run time equation of this code and explain all the terms.

We take \\(n\\) to be the number of nodes in the tree.

\\(T(n)=\begin{cases}
    c_0 & \text{if $n<2$}.\\\\
    c_1 + 2T(n/2) & \text{otherwise}.
  \end{cases}\\)
  
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​

\\(T(n) = 2T(n/2) + c_1\\)
$$
\begin{gather}
a = 2, \quad b = 2, \quad f(n) = \Theta(1) \\\\
n^{log_b(a)} = n^{log_2 (2)} = n^1 \\\\
f(n) = \mathcal{O}(n), \\\\
\text{thus Case 1 of the master method, the work is dominated by the leaves} \\\\
T(n) = \Theta(n)
\end{gather}
$$