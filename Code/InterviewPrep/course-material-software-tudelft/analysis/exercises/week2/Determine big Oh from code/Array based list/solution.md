In the worst case, the `binary_search` method will search the entire array and only stop when `low = high - 1`.
As \\(n = high - low\\), the algorithm stops when \\(n = 1\\).

####1) Corresponding recurrence equation, base case and explanation of all terms

\\(T(n)=\begin{cases}
    c_0 & \text{if $n=1$}.\\\\
    c_1 + T(n/2) & \text{otherwise}.
  \end{cases}\\)

Where:

- \\(n\\) is the value as described, `high - low`
- \\(c_0\\) represents the constant-time operations on line 4 and 5
- \\(c_1\\) represents the constant-time operations on line 4, 6-8 and 10 + 11 or 10 + 12 + 13

####2) Derivation of the closed form solution

\\(T(n) = c_1 + T(n/2)\\)

Fill in \\(T(n/2)\\):

\\(T(n) = c_1 + c+1 + T(n/4)\\)

\\(T(n) = 2c_1 + T(n/4)\\)

Fill in \\(T(n/4)\\):

\\(T(n) = 3c_1 + T(n/8)\\)

For \\(k\\) repetitions of substitution:

\\(T(n) = kc_1 + T(n/2^k)\\)

Take \\(k = log_2 n\\):

\\(T(n) = c_1 log_2 n + T(n/n)\\)

\\(T(n) = c_1 log_2 n + T(1)\\)

\\(T(n) = c_1 log_2 n + c_0\\)

####3) Simplification and final computational complexity in terms of Big Oh notation.

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = log_2 n\\), which is a logarithmic growth. Thus, the computational complexity is \\(O(log_2 n)\\).
