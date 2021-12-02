\\(f(n)\\) is \\(\Omega(g(n))\\) iff there is a real constant \\(c > 0\\) and an integer constant \\(n_0 \geq 1\\) such that \\(f(n) \geq c g(n)\\), for all \\(n \geq n_0\\).

We will solve \\(f(n) \geq c g(n)\\) by substituting the definitions of \\(f(n)\\) and \\(g(n)\\):
$$
3n^{4} + 21n^{2} + 10 \geq c(n^{3} + 17) \\\\
3n^{4} - c \cdot n^{3} + 21n^{2} - 17c \geq -10 \\\\
$$
There are many possible values for \\(c\\) and \\(n_0\\), for example \\(c=1\\) and \\(n_0 = 1\\).
For these constants, the inequality holds: \\(3 - 1 + 21 - 17 \geq -10\\)
Since the term containing \\(n^4\\) will grow as \\(n\\) increases, the left hand side will be greater than \\(-10\\) for all \\(n \geq n_0\\).
