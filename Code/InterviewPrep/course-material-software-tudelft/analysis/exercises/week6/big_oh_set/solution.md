#### 1) Run time equation:

\\(T(n) = c_0 + nc_1 + n(c_2 + \mathcal{O}(n)) \\)

Where:

- \\(n\\) corresponds to the length of `xs`.
- \\(c_0\\) corresponds to the constant-time operations to initialize `ys`, `zs` and `res`.
- \\(c_1\\) corresponds to the (amortized) constant-time additions and arithmetic operations to create the items in `ys` and `zs`. 
- \\(c_2\\) corresponds to getting the next `x` in line 10 and the constant-time operations in line 11.
- \\(\mathcal{O}(n)\\) corresponds to checking the condition on line 12 and the adding of `x` to `res` on line 13.

Note that in the worst case `xs` contains no duplicate elements, therefore `len(xs) == len(ys) == len(zs)`.
Furthermore the intersection function has a much better expected time complexity (\\(\mathcal{O}(n)\\)), the worst case is highly unlikely to occur. The `in` and `add` functions of sets have an expected complexity of \\(\mathcal{O}(1)\\), but may need linear time if all elements have colliding hash values.

#### 2) Run time complexity in terms of Big Oh notation

After simplification, the constant terms and factors can be disregarded as \\(n\\) grows to infinity. This leaves us with \\(T(n) = n + n + n^2\\). The fastest growing term in the equation is \\(n^2\\), which is quadratic growth. Therefore the run time complexity is \\(\mathcal{O}(n^2)\\)
