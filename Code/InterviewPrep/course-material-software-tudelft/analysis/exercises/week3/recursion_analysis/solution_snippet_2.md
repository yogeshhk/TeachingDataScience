\\(T(n)=\begin{cases}
    c_0 & \text{if $n=1$}.\\\\
    T(n-1) + n \cdot c_1 + c_2 & \text{otherwise}.
  \end{cases}\\)
  
Where:
* \\(n\\) is the size of the list `xs`
* \\(c_0\\) represents the constant-time operations on lines 5 and 6, when `len(xs) == 1`
* \\(n \cdot c_1\\) represents the operations needed for adding all elements but the first one to a new list `ys`
* \\(c_2\\) represents the constant-time operations for checking the condition, initializing an empty list and doing the computations on line 10

By repeatedly unfolding we can find the closed form:

We fill in \\(T(n - 1)\\), where \\(T(n - 1) = T(n - 2) + n \cdot c_1 + c_2\\)
$$
T(n) = T(n - 2) + n \cdot c_1 +  c_2 + n \cdot c_1 + c_2 \\\\
T(n) = T(n - 2) + 2n \cdot c_1 + 2 \cdot c_2 \\\\
$$

We fill in \\(T(n - 2)\\), where \\(T(n - 2) = T(n - 3) + n \cdot c_1 + c_2 \\)
$$
T(n) = T(n - 3) + n \cdot c_1 + c_2 + 2n \cdot c_1 + 2 \cdot c_2 \\\\
T(n) = T(n - 3) + 3n \cdot c_1 + 3 \cdot c_2 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(n) = T(n - k) + k \cdot n \cdot c_1 + k \cdot c_2
$$

To substitute out the recursive part, \\(T(n - k)\\), from the equation, we need to make \\(T(n - k )\\) equal to the base case, \\(T(0)\\).
Which means that we need to solve the equation \\(n - k = 0\\) for \\(k\\).

$$
n - k = 0 \\\\
k = n \\\\
$$

So we can substitute \\(k\\) with \\(n\\).

$$
T(n) = T(n - n) + n^{2} \cdot c_1 + n \cdot c_2 \\\\
T(n) = T(0) + n^{2} \cdot c_1 + n \cdot c_2 \\\\
T(n) = c_0 + n^{2} \cdot c_1 + n \cdot c_2 \\\\
$$

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n^{2} + n\\), which is a quadratic growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(n^{2})\\).