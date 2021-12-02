In the worst case, the `search` method will search the entire list and only stop when `high < low`.
As \\(n = high - low\\), the algorithm stops when \\(n = -1\\) (when the length of the list is even) or \\(n = -2\\) (when the length of the list is odd).

####1) Corresponding recurrence equation, base case and explanation of all terms

\\(T(n)=\begin{cases}
    c_0 & \text{if $n=-1$ or $n=-2$}.\\\\
    c_1 + T(n-2) & \text{otherwise}.
  \end{cases}\\)

Where:

- \\(n\\) is the value as described, `high - low`
- \\(c_0\\) represents the constant-time operations on line 4 and 5
- \\(c_1\\) represents the constant-time operations on line 4, 6 and 7

####2) Derivation of the closed form solution, in the case that \\(n\\) is even

\\(T(n) = T(n-2) + c_1\\)

\\(T(n-2) = T(n-4) +c_1\\)

\\(T(n) = T(n-2k) + kc_1\\)

Take \\(k=n/2+1\\):

\\(T(n) = T(-2) + \frac{nc_1}{2} + c_1 = c_0 + \frac{nc_1}{2} + c_1\\)

####3) Simplification and final computational complexity in terms of Big Oh notation.

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n\\), which is a linear growth. Thus, the computational complexity is \\(O(n)\\).
