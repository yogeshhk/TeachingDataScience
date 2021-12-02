\\(T(n)=\begin{cases}
    c_0, & \text{if $n=0$ or $n=1$}.\\\\
    T(n) = c_1 + T(n-2) & \text{otherwise}.
  \end{cases}\\)
  
Where:
* \\(n\\) is \\(h - l + 1\\), which is the length of the implicit sublist
* \\(c_0\\) represents the constant-time operations on line 1, 2, and 6
* \\(c_1\\) represents the constant-time operations on line 6, 7 and 8

We fill in \\(T(n-2)\\)

\\(T(n) = c_1 + c_1 + T(n-4)\\)

\\(T(n) = 2 \cdot c_1 + T(n-4)\\)

We fill in \\(T(n-4)\\)

\\(T(n) = 3 \cdot c_1 + T(n-6)\\)

For k repetitions of substitution:

\\(T(n) = k \cdot c_1 + T(n-2k)\\)

Take \\(k = \frac{n}{2}\\) to arrive at the base case:

\\(T(n) = \frac{n \cdot c_1}{2} + T(0)\\)

\\(T(n) = \frac{n \cdot c_1}{2} + c_0\\)

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n\\).
This is linear growth, therefore the computational complexity is \\(O(n)\\).
