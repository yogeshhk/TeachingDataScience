- Equation 1 Solution:
\\(T(n) = 3T(n/2) + n^2log(n)\\)
$$
\begin{gather}
a = 3, \quad b = 2, \quad f(n) = n^2log(n) \\\\
n^{log_b(a)} = n^{log_2(3)} = n^{1...} \\\\
f(n) = \Omega(n^{1.... + e}) \quad e > 0, \\\\
\text{thus Case 3, the work is dominated by the root} \\\\
T(n) = \Theta(n^2log(n)) \\\\
\text{We have to check for the regularity condition.} \\\\
\text{Since \\(n^2log(n)\\) is polynomial it holds or we can prove it:} \\\\
af(n/b) \leq cf(n) \quad c < 1 \\\\
3(n/2)^2log(n/2) \leq cn^2log(n) \\\\
3/4n^2log(n/2) \leq cn^2log(n) \\\\
\text{we can take} \quad c = 3/4 \text{ as } log(n/2) < log(n)
\end{gather}
$$


- Equation 2 Solution:
\\(T(n) = 8T(n/2) + n^3\\)
$$
\begin{gather}
a = 8, \quad b = 2, \quad f(n) = n^3 \\\\
n^{log_b(a)} = n^{log_2(8)} = n^{3} \\\\
f(n) = \Theta(n^{3}), \\\\
\text{thus Case 2, the work is evenly split amongst leaves and root} \\\\
T(n) = \Theta(n^{3}log(n))
\end{gather}
$$

