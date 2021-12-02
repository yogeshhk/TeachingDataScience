####1) Corresponding recurrence equation, base case and explanation of all terms

\\(T(n)=\begin{cases}
    c_0 & \text{if $n<2$}.\\\\
    c_1 + nc_2 + 2T(n/2) & \text{otherwise}.
  \end{cases}\\)


Where:

- \\(n\\) is the length of the list `xs`
- \\(c_0\\) represents the constant-time operations on line 4 and 5
- \\(c_1\\) represents the constant-time operations on lines 4, 6-8, 11, 14, 15 and the return on line 23
- \\(c_2\\) represents the constant-time operations on lines 16-22 and the list extension on line 23

####2) Run time complexity with the master method.

$$
T(n) = c_1 + nc_2 + 2T(n/2)\\\\
a = 2, \quad b = 2, \quad f(n) = n \\\\
n^{log_b(a)} = n^{log_2(2)} = n^1 \\\\
f(n) = \Theta(n)$$
Thus this is case 2, the work is evenly split amongst leaves and root.
$$T(n) = \Theta(n log_2 n))$$

-----

####2) Derivation of the closed form solution with recurrence equations.

##### We can use repeated unfolding to derive a closed form, but this is much more work than using the master method.

\\(T(n) = c_1 + nc_2 + 2T(n/2)\\)

Fill in \\(T(n/2)\\):

\\(T(n) = c_1 + nc_2 + 2(c_1 + \frac{nc_2}{2} + 2T(n/4))\\)

\\(T(n) = 3c_1 + 2nc_2 +  4T(n/4)\\)

Fill in \\(T(n/4)\\):

\\(T(n) = 3c_1 + 2nc_2 +  4(c_1 + \frac{nc_2}{4} + 2T(n/8))\\)

\\(T(n) = 7c_1 + 3nc_2 +  8T(n/8)\\)

For \\(k\\) repetitions of substitution:

\\(T(n) = (2^k-1)c_1 + knc_2 +  2^kT(n/2^k)\\)

Let \\(k = log_2 n\\) to arrive at the base case:

\\(T(n) = (n-1)c_1 + log_2 n n c_2 +  nT(n/n)\\)

\\(T(n) = (n-1)c_1 + log_2 n n c_2 +  nT(1)\\)

\\(T(n) = (n-1)c_1 + log_2 n n c_2 +  nc_0\\)

##### Simplification and final run time complexity

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n log_2 n + n \\).
The fastest growing term is \\(n log_2 n\\), which is linearithmic growth. Thus, the run time complexity is \\(O(n log_2 n)\\).
