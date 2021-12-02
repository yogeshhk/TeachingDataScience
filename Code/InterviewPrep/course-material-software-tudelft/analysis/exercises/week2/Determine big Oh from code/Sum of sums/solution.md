####1) Corresponding recurrence equation, base case and explanation of all terms

\\(T(n)=\begin{cases}
    c_0 & \text{if $n=0$}.\\\\
    c_1 + nc_2 + T(n-1) & \text{otherwise}.
  \end{cases}\\)

- \\(n\\) is the value of `n`
- \\(c_0\\) represents the constant-time operations on lines 2 and 3
- \\(c_1\\) represents the constant-time operations on lines 2 and 6
- \\(c_2\\) represents the constant-time operations on lines 4 and 5, which are executed \\(n\\) times

####2) Derivation of the closed form solution

\\(T(n) = c_1 + nc_2 + T(n-1)\\)

We fill in \\(T(n-1)\\):

\\(T(n) = c_1 + nc_2 + c_1 + (n - 1) c_2 + T(n-2)\\)

\\(T(n) = 2c_1 + (n + n - 1) c_2 + T(n-2)\\)

We fill in \\(T(n-2)\\):

\\(T(n) = 2c_1 + (n + n - 1) c_2 + c_1 + (n - 2) c_2 + T(n-3)\\)

\\(T(n) = 3c_1 + (n + n - 1 + n - 2) c_2 + T(n-3)\\)

For \\(k\\) repetitions of substition:

\\(T(n) = kc_1 + (n + n - 1 + ... + n - (k-1)) c_2 + T(n-k)\\)

Let \\(n = k\\):

\\(T(n) = nc_1 + (n + n - 1 + ... + n - (n-1)) c_2 + T(n-n)\\)

\\(T(n) = nc_1 + (n + n - 1 + ... + 1) c_2 + T(0)\\)

\\(T(n) = nc_1 + \frac{c_2n(n+1)}{2} + c_0\\)

####3) Simplification and final computational complexity in terms of Big Oh notation.

After simplification, the constant terms and factors are removed.
This leaves us with \\(T(n) = n + n^2\\). The largest growing term is \\(n^2\\).
Thus, the computational complexity is \\(O(n^2)\\).
