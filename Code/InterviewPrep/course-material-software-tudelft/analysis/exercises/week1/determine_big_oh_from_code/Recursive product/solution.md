\\(T(n)=\begin{cases}
    c_0 & \text{if $n=0$}.\\\\
    c_1 + T(n-1) & \text{otherwise}.
  \end{cases}\\)

Where:
* \\(n\\) is the absolute value of `a`
* \\(c_0\\) represents the constant-time operations on line 2 and 3
* \\(c_1\\) represents the constant-time operations on line 2, 4 and 5 or 6

We fill in \\(T(n-1)\\)

\\(T(n) = c_1 + c_1 + T(n-2)\\)

\\(T(n) = 2 \cdot c_1 + T(n-2)\\)

We fill in \\(T(n-2)\\)

\\(T(n) = 3 \cdot c_1 + T(n-3)\\)

For k repetitions of substitution:

\\(T(n) = k \cdot c_1 + T(n-k)\\)

Take \\(k = n\\) to arrive at the base case:

\\(T(n) = n \cdot c_1 + T(0)\\)

\\(T(n) = n \cdot c_1 + c_0\\)

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n\\).
This is linear growth, therefore the computational complexity is \\(O(n)\\).
