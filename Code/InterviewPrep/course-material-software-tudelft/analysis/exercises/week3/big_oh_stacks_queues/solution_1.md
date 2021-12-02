####1) Corresponding run time equation for method_x and explanation of all terms

\\(T(n)=c_0 + nc_1 + nc_2\\)

Which is equal to:

\\(T(n)=c_0 + nc_3\\)
\\(where \quad c_3 = c_1 + c_2\\)

Where:

- \\(n\\) is the size of the stack `s`
- \\(c_0\\) represents the constant-time operations on lines 3, 4, 6, 7 (declare temporary stack, initialize the first and second loop, push given element)
- \\(c_1\\) represents the constant-time operations on lines 5 (pop the top value from Stack `s` and push it to Stack `temp`)
- \\(c_2\\) represents the constant-time operations on lines 8 (pop the top value from Stack `temp` and push it to Stack `s`)

####2) State the run time complexity in terms of Big Oh notation.

After simplification, the constant terms and factors can be disregarded as n grows to infinity. This leaves us with \\(T(n) = n\\).
This is linear growth. Thus, the run time complexity is \\(\mathcal{O}(n)\\).


