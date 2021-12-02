\\(f(n)\\) is \\(\Theta(g(n))\\) iff there are real constants \\(c > 0\\) and \\(c' > 0\\), and and an integer constant \\(n_0 \geq 1\\) such that \\(c g(n) \leq f(n) \leq c' g(n)\\), for all \\(n \geq n_0\\).

We will solve \\(c g(n) \leq f(n) \leq c' g(n)\\) in two parts, as this expression is equivalent to the two inequalities \\(c g(n) \leq f(n)\\) and \\(f(n) \leq c' g(n)\\).

1. We will solve \\(c g(n) \leq f(n)\\) by substituting the definitions of \\(f(n)\\) and \\(g(n)\\):
$$
2cn^2 \leq 3n (4n + \sqrt{n}) \\\\
2cn^2 \leq 12n^2 + 3n\sqrt{n} \\\\
(2c - 12)n^2 \leq 3n\sqrt{n} \\\\
(2c - 12)\sqrt{n} \leq 3
$$

2. We will solve \\(f(n) \leq c' g(n)\\) by substituting the definitions of \\(f(n)\\) and \\(g(n)\\):
$$
3n (4n + \sqrt{n}) \leq 2c'n^2 \\\\
12n^2 + 3n\sqrt{n} \leq 2c'n^2 \\\\
(12 - 2c')n^2 \leq -3n\sqrt{n} \\\\
(12 - 2c')\sqrt{n} \leq -3
$$

There are many possible values for \\(c\\), \\(c'\\), and \\(n_0\\), for example \\(c = 5\\), \\(c' = 7\\) and \\(n_0 = 4\\).
For these constants, the two inequalities hold for all \\(n \geq n_0\\):

1. \\((10 - 12)\cdot 2 \leq 3\\)
    Since the term containing \\(\sqrt{n}\\) will shrink as \\(n\\) increases, the left hand side will be less than \\(3\\) for all \\(n \geq n_0\\).
2. \\((12 - 14)\cdot 2 \leq -3\\)
    Since the term containing \\(\sqrt{n}\\) will shrink as \\(n\\) increases, the left hand side will be less than \\(-3\\) for all \\(n \geq n_0\\).

With this, we have proven that \\(f(n)\\) is \\(\Theta(g(n))\\).
