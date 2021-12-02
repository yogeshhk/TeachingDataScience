\\(T(n) = 2T(n/2) + n/2\\)
$$
\begin{gather}
a = 2, \quad b = 2, \quad f(n) = n/2 \\\\
n^{\log_b(a)} = n^{\log_2(2)} = n \\\\
f(n) = \Theta(n), \\\\
\text{thus Case 2, the work is evenly split amongst leaves and root} \\\\
T(n) = \Theta(n\log(n))
\end{gather}
$$
