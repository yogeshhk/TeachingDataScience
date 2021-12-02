- Equation 1 Solution:
\\(T(n) = 3T(n/2) + n\\)

\begin{gather}
a = 3, \quad b = 2, \quad f(n) = n \\\\
n^{log_b(a)} = n^{log_2(3)} = n^{1.5...} \\\\
f(n) = \mathcal{O}(n^{1.5...}), \\\\
\text{thus Case 1, the work is dominated by the leaves} \\\\
T(n) = \Theta(n^{log_b(a)})
\end{gather}


- Equation 2 Solution:
\\(T(n) = T(n/7) + n\\)

\begin{gather}
a = 1, \quad b = 7, \quad f(n) = n \\\\
n^{log_b(a)} = n^{log_7(1)} = n^{0} \\\\
f(n) = \Omega(n^{0 + e}) \quad e > 0, \\\\
\text{thus Case 3, the work is dominated by the root} \\\\
T(n) = \Theta(n) \\\\
\text{We have to check for the regularity condition.} \\\\
\text{Since n is polynomial it holds or we can prove it:} \\\\
af(n/b) \leq cf(n) \quad c < 1 \\\\
n/7 \leq 1/7*n \\\\
\text{Which means that for \\(c = 1/7\\) the regularity condition holds}
\end{gather}

- Equation 3 Solution:
\\(T(n) = 2T(n/2) + nlog(n)\\)

\begin{gather}
a = 2, \quad b = 2, \quad f(n) = nlog(n) \\\\
n^{log_b(a)} = n^{log_2(2)} = n \\\\
\text{But \\(f(n)\neq \mathcal{O}(n^{1-e}) for \quad e >0 \\)} \\\\
\text{and \\(f(n)\neq \Theta(n)\\)} \\\\
\text{and \\(f(n)\neq \Omega(n^{1+e}) for \quad e > 0\\)} \\\\
\text{as \\(n^c = \Omega(log(n))\\) thus \\(n^{c+e} = \Omega(nlog(n))\\)} \\\\
\text{So this equation cannot be solved using the Master method,}\\\\
\text{as it doesn't belong to any of its cases.}
\end{gather}


- Equation 4 Solution:
\\(T(n) = 9T(n/3) + n^2\\)

\begin{gather}
a = 9, \quad b = 3, \quad f(n) = n^2 \\\\
n^{log_b(a)} = n^{log_3(9)} = n^{2} \\\\
f(n) = \Theta(n^{2}), \\\\
\text{thus Case 2, the work is evenly split amongst leaves and root} \\\\
T(n) = \Theta(n^{2}log(n))
\end{gather}


- Equation 5 Solution:
\\(T(n) = \sqrt2T(n/2) + log(n)\\)

\begin{gather}
a = \sqrt2, \quad b = 2, \quad f(n) = log(n) \\\\
n^{log_b(a)} = n^{log_2(\sqrt2)} = n^{0.5} \\\\
f(n) = \mathcal{O}(n^{0.5}), \\\\
\text{thus Case 1, the work is dominated by the leaves} \\\\
T(n) = \Theta(\sqrt{n})
\end{gather}