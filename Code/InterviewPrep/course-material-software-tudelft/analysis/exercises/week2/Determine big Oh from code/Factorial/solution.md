####1) Corresponding recurrence equation, base case and explanation of all terms

\\(T(n)=\begin{cases}
    c_0 & \text{if $n=0$}.\\\\
    c_1 + T(n-1) & \text{otherwise}.
  \end{cases}\\)

Where:

* \\(n\\) is the value of `n`
* \\(c_0\\) represents the constant-time operations on line 2 and 3
* \\(c_1\\) represents the constant-time operations on line 2, 4 and 5

####2) Derivation of the closed form solution

\\(T(n) = c_1 + T(n-1)\\)

We fill in \\(T(n-1)\\):

\\(T(n) = c_1 + c_1 + T(n-2)\\)

We fill in \\(T(n-2)\\):

\\(T(n) = c_1 + c_1 + c_1 + T(n-3)\\)

For \\(k\\) substitutions:

\\(T(n) = kc_1 + T(n-k)\\)

Choose \\(k = n\\):

\\(T(n) = nc_1 + T(0)\\)

\\(T(n) = nc_1 + c_0\\)


####3) Simplification and final computational complexity in terms of Big Oh notation.

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n\\), which is a linear growth. Thus, the computational complexity is \\(O(n)\\).
