\\(T(n) = c_0 + n \cdot c1 + n^2 \cdot c_2\\) where:
* \\(n\\) is the size of the sets
* \\(c_0\\) represents the constant-time operations on lines 2 and 6
* \\(c_1\\) represents the constant-time operations that select the next `x` for the next iteration on line 3
* \\(c_2\\) represents the constant-time operations that select the next y for the next iteration on line 4,
and the constant-time operations on line 5

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n + n^2\\).
The largest growing term is \\(n^2\\), which is a quadratic growth. Thus, the computational complexity is \\(O(n^2)\\).
