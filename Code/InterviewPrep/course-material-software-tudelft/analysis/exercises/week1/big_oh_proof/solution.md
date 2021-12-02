\\(f(n)\\)  is \\(\mathcal{O}(g(n))\\) iff there is a real constant \\(c > 0\\) and an integer constant \\( n_0 \geq 1 \\) such that \\( f(n) \leq cg(n) \\), for all \\(n \geq n_0\\).

We will solve \\( f(n) \leq cg(n) \\) by substituting the definitions of \\(f(n)\\) and \\(g(n)\\):

$$
\begin{align}
12n + 7 &\leq cn \\\\
7 &\leq cn - 12n \\\\
7 &\leq n(c - 12)
\end{align}
$$

There are many possible values for \\(c\\) and \\(n_0\\), for example \\(c = 19 \\) and \\(n_0 = 1\\).
For these constants, the inequality holds: \\(7 \leq (19-12)=7\\)
This is true for any positive \\(n\\), because by increasing \\(n\\) the inequality \\(7 \leq n(c - 12)\\) 
still holds, so surely it holds for all \\(n \geq n_0 = 1\\).
