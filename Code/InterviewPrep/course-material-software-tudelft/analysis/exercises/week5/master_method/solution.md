\\(T(n) = 4T(n/2) + n^2 + n\\)
$$
\begin{gather}
a = 4, \quad b = 2, \quad f(n) = n^2 + n \\\\
n^{\log_b(a)} = n^{\log_2(4)} = n^2 \\\\
f(n) = \Theta(n), \\\\
\text{thus Case 2, the work is evenly split amongst leaves and root} \\\\
T(n) = \Theta(n^2\log(n))
\end{gather}
$$