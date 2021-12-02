####1) Corresponding run time equation and explanation of all terms

\\(T(n)=c_0 + nc_1 + \sum\limits_{i=0}^{n} ic_2\\)

Which is equal to:

\\(T(n)=c_0 + nc_1 + \frac{n(n+1)c_2}{2}\\)

Where:

- \\(n\\) is the size of the list `xs`
- \\(c_0\\) represents the constant-time operations on line 4 (initialize the first loop)
- \\(c_1\\) represents the constant-time operations on lines 4 (get the next value for `x`) and 5 (initialize the second loop)
- \\(c_2\\) represents the constant-time operations on lines 5 (get the next value for `y`), 6 and 7

####2) State the run time complexity in terms of Big Oh notation.

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n + n^2\\).
The fastest growing term is \\(n^2\\), which is quadratic growth. Thus, the run time complexity is \\(O(n^2)\\).
