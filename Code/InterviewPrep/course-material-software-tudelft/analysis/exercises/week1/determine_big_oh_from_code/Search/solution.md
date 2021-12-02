\\(T(n) = c_0 + n \cdot c_1\\) where:
* \\(n\\) is the length of `xs`
* \\(c_0\\) is the constant time to execute line 4 or 5 (one of the two is executed)
* \\(c_1\\) represents the constant-time operations in line 2 and 3

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n)=n\\), which is a linear growth.
Thus, the computational complexity is \\(O(n)\\).
