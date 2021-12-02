Note that line 11 - 14 are bubble sort. For which we know that the run time equation can be written as: \\(T(n)=c_0 + nc_1 + \frac{n(n+1)c_2}{2}\\)

Combining that with the rest of the code snippet gives us:

$$
T(n) = c_2 + c_3 \cdot n + c_4 \cdot n^{2} + n \cdot (c_0 + nc_1 + \frac{n(n+1)c_2}{2})
$$

Where:
- \\(n\\): The length of the list `xs`
- \\(c_0\\): The constant amount of operations on line 10 (initializing the for-loop)
- \\(c_1\\): The constant amount of operations on line 11 (initializing the for-loop, incrementing `i`)
- \\(c_2\\): The constant amount of operations required for initializing an empty list, initializing `sz` and returning the result
- \\(c_3\\): The constant amount of operations required for getting an element from the list `ys` at position `it`, appending an element to the end of the resulting list, and incrementing `it`
- \\(c_4\\): The constant amount of operations required for copying the list `xs` to `ys`

After simplification, the constant term and factors are removed. This leaves us with \\(T(n) = n + n^{2} + n^{3}\\). 
The fastest growing term is \\(n^{3}\\), which is cubic growth. Thus, the run time complexity is \\(\mathcal O(n^{3}\\)).