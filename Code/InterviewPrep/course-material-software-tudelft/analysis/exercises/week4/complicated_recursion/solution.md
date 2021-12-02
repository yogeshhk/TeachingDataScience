#### Recursive run time equation:

\\(n\\) corresponds to parameter \\(n\\).
Recurrence equation looks as follows:
\\(T(n) = 2T(n/5) + c_1\sqrt{n} + c_2 \\)
and the corresponding base case equation
\\(T(0)=c_3\\)

- In the recursive case:
    - The body of the loop on lines 9 and 10 takes \\(c_1\\) time and runs \\(\sqrt{n}\\) times.
    - Performing a comparison (line 4), arithmetic (lines 6,7,9,11) and assignments (lines 6,7,8) takes constant time \\(c_2\\).
    - Performing two recursive calls with (more or less) \\(n / 5\\) as argument, takes \\(2T(n/5)\\) time.
- In the base case:
    - \\(c_3\\) corresponds to the constant amount of time for performing the comparison and returning `0`.

#### Big-Oh notation:

In this case, we can use the Master Method.

$$
\begin{gather}
a = 2, \quad b = 5, \quad f(n) = \sqrt{n} \\\\
n^{log_b(a)} = n^{log_5(2)} \\\\
f(n) = n^{1/2} = \Omega(n^{log_5(2) + \varepsilon}) \quad \varepsilon > 0, \\\\
\text{thus Case 3, the work is dominated by the root} \\\\
T(n) = \Theta(\sqrt{n}) \\\\
\text{We have to check for the regularity condition.} \\\\
\text{Since n is polynomial, it holds, or we can prove it:} \\\\
af(n/b) \leq cf(n), \quad c < 1 \\\\
2\sqrt{n/5} \leq c\cdot \sqrt{n} \\\\
\sqrt{4/5}\cdot\sqrt{n} \leq c\cdot \sqrt{n} \\\\
\text{Which means that for \\(c = \sqrt{4/5} < 1\\), the regularity condition holds.}
\end{gather}
$$
