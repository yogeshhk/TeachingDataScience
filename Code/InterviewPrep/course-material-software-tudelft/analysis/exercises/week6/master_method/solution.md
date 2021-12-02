\\(T(n) = 6T(n/2) + n^2 \log(n)\\)
$$
\begin{gather}
a = 6, \quad b = 2, \quad f(n) = n^2 \log(n) \\\\
n^{\log_b(a)} = n^{\log_2(6)} = n^{2...} \\\\
f(n) = \mathcal{O}(n^{\log_2(6)}), \\\\
\text{thus Case 1, the work is dominated by the leaves} \\\\
T(n) = \Theta(n^{\log_b(a)}) = \Theta(n^{\log_2(6)})
\end{gather}
$$