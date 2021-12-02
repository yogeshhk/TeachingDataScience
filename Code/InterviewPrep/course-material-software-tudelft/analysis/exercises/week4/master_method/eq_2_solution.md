\\(T(n) = 4T(n/2) + n/\log(n)\\)
$$
\begin{gather}
a = 4, \quad b = 2, \quad f(n) = n/\log(n) \\\\
n^{\log_b(a)} = n^{\log_2(4)} = n^{2} \\\\
f(n) = \mathcal{O}(n^{2}), \\\\
\text{thus Case 1, the work is dominated by the leaves} \\\\
T(n) = \Theta(n^{\log_b(a)})
\end{gather}
$$
