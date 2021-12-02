#### Recursive run time equation:

\\(n\\) corresponds to parameter `n`.
The recurrence equation looks as follows:
$$
T(n) = \begin{cases}
c_3 & \text{if } n \leq 0 \\\\
nT(n-1)+c_1n+c_2 & \text{otherwise} 
\end{cases}
$$

- In the recursive case:
    - The body of the loop on lines 4 and 5 takes \\(c_1\\) time and runs \\(n\\) times.
    - Performing a variable assignment (line 2), a comparison (line 3), initializing the loop (line 4), and a return (line 6) take constant time \\(c_2\\).
    - Performing \\(n\\) recursive calls with \\(n - 1\\) as first argument, takes \\(nT(n-1)\\) time.
- In the base case:
    - \\(c_3\\) corresponds to the constant amount of time for performing a variable assignment (line 2), a comparison (line 3) and returning `res` (line 6).

####Deriving the closed form:

$$
\begin{align}
T(n)&=nT(n-1)+c_1n+c_2 \\\\
T(n)&=n((n-1)T(n-2)+c_1(n-1)+c_2) + c_1n + c_2 \\\\
T(n)&=n(n-1)T(n-2) + c_1n(n-1)+c_1n + c_2n+c_2 \\\\
T(n)&=n(n-1)((n-2)T(n-3)+c_1(n-2)+c_2) + c_1n(n-1)+c_1n + c_2n+c_2 \\\\
T(n)&=n(n-1)(n-2)T(n-3) + c_1n(n-1)(n-2)+c_1n(n-1)+c_1n + c_2n(n-1)+c_2n+c_2 \\\\
    &\ldots \text{After $k$ iterations} \ldots \\\\
T(n)&=T(n-k)\prod_{i=n-k+1}^n i + c_1\sum_{j=1}^k\prod_{i=n-j+1}^n i + c_2\sum_{j=1}^k\prod_{i=n-j+2}^n i \\\\
    &\text{ (take $n-k=0 \quad\to\quad k=n$)} \\\\
T(n)&=T(0)\prod_{i=1}^n i + c_1\sum_{j=1}^n\prod_{i=n-j+1}^n i + c_2\sum_{j=1}^n\prod_{i=n-j+2}^n i \\\\
T(n)&=c_3n! + c_1\sum_{j=1}^n\frac{n!}{(n-j)!} + c_2\sum_{j=1}^n\frac{n!}{(n-j+1)!} \\\\
\end{align}
$$

####Simplification:
$$
\begin{align}
T(n)&=c_3n! + c_1\sum_{j=1}^n\frac{n!}{(n-j)!} + c_2\sum_{j=1}^n\frac{n!}{(n-j+1)!} \\\\
    &\leq c_3n! + c_1n\cdot n! + c_2n\cdot n! \\\\
    &= c_3n! + (c_1 + c_2)n\cdot n! \\\\
\end{align}
$$

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n! + n\cdot n!\\). Since \\(n \cdot n!\\) grows faster than \\(n!\\) on large inputs, the complexity in Big-Oh notation is \\(\mathcal{O}(n \cdot n!)\\).
