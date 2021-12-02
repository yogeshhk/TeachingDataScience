####1) Corresponding recurrence equation, base case and explanation of all terms

\\(T(n)=\begin{cases}
    c_0 & \text{if $n=-1$}.\\\\
    c_1 + nc_2 + T(n-1) & \text{otherwise}.
  \end{cases}\\)


Where:

- \\(n\\) is `len(xs) - x`
- \\(c_0\\) represents the constant-time operations on line 4
- \\(c_1\\) represents the constant-time operations on lines 4, 5. 6 (to initialize the loop), 9 and 10
- \\(c_2\\) represents the constant-time operations on lines 6 (get the next value for `y`), 7 and 8

####2) Derivation of the closed form solution

\\(T(n) = c_1 + nc_2 + T(n-1)\\)

Fill in \\(T(n-1)\\):

\\(T(n) = c_1 + nc_2 + c_1 + (n-1)c_2 + T(n-2)\\)

\\(T(n) = 2c_1 + (n + n-1)c_2 + T(n-2)\\)

Fill in \\(T(n-2)\\):

\\(T(n) = 2c_1 + (n + n-1)c_2 + c_1 + (n-2)c_2 + T(n-3)\\)

\\(T(n) = 3c_1 + (n + n-1 + n-2)c_2 + T(n-3)\\)

For \\(k\\) repetitions of substitution:

\\(T(n) = kc_1 + (n + n-1 + ... + n-k)c_2 + T(n-k)\\)

Take \\(k = n + 1\\) to arrive at the base case:

\\(T(n) = (n+1)c_1 + (n + n-1 + ... + n-(n+1))c_2 + T(n-(n+1))\\)

\\(T(n) = (n+1)c_1 + (n + n-1 + ... - 1)c_2 + T(-1)\\)

\\(T(n) = (n+1)c_1 + \frac{n(n+1)c_2}{2} - c_2 + c_0\\)

\\(T(n) = (n+1)c_1 + \frac{(n+n^2+2)c_2}{2} + c_0\\)

####3) Simplification and final run time complexity in terms of Big Oh notation.

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n + n^2\\).
The fastest growing term is \\(n^2\\), which is quadratic growth. Thus, the run time complexity is \\(O(n^2)\\).
