\\(T(n) = c_0 + n \cdot c_1 + n^2 \cdot c_2 + n^3 \cdot c_3\\) where:

* \\(n\\) represents the dimension of the matrices (height or width)
* \\(c_0\\) represents the constant-time operations on line 2 and 11
* \\(c_1\\) represents the constant-time operations on line 3, 4 and 10
* \\(c_2\\) represents the constant-time operations on line 5, 6 and 9
* \\(c_3\\) represents the constant-time operations on line 7 and 8

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n)=n+n^2+n^3\\).
The largest growing term is \\(n^3\\), therefore the computational complexity is \\(O(n^3)\\).
