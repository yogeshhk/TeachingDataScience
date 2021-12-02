\\(f(n)\\) is \\(\Omega(g(n))\\) iff there is a real constant \\(c > 0\\) and an integer constant \\(n_0 \geq 1\\) such that \\(f(n) \geq c g(n)\\), for all \\(n \geq n_0\\).

We will solve \\(f(n) \geq c g(n)\\) by substituting the definitions of \\(f(n)\\) and \\(g(n)\\):
$$
6n^2 + 7n + 42 \geq c(42n + 6) \\\\
6n^2 + 7n - 42cn - 6c \geq -42 \\\\
6n^2 + (7 - 42c)n - 6c \geq -42
$$
There are many possible values for \\(c\\) and \\(n_0\\), for example \\(c=\frac16\\) and \\(n_0 = 1\\).
For these constants, the inequality holds: \\(6 + (7 - 7) - 1 = 5 \geq -42\\)
Since the term containing \\(n^2\\) will grow as \\(n\\) increases, the left hand side will be greater than \\(-42\\) for all \\(n \geq n_0\\).
