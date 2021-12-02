\\(T(n)=\begin{cases}
    c_0 & \text{if $n<=1$}.\\\\
    2 \cdot T\left(\frac{n}{2}\right) + n \cdot c_1 + c_2 & \text{otherwise}.
  \end{cases}\\)
  
Where:
* \\(n\\) is the size of the list `xs`
* \\(c_0\\) represents the constant-time operations on lines 5 and 6, when `len(xs) == 1`
* \\(n \cdot c_1\\) represents the operations needed to create the two sublists, `ys` and `zs`.
* \\(c_2\\) represents the constant-time operations for checking the condition, computing the middle index and the computation on line 10

By repeatedly unfolding we can find the closed form:

We fill in \\(T\left(\frac{n}{2}\right)\\), where \\(T\left(\frac{n}{2}\right) = 2 \cdot T\left(\frac{n}{4}\right) + n \cdot c_1 + c_2\\)
$$
T(n) = 2 \cdot (2 \cdot T\left(\frac{n}{4}\right) + n \cdot c_1 + c_2) + n \cdot c_1 + c_2 \\\\
T(n) = 4 \cdot T\left(\frac{n}{4}\right) + 3n \cdot c_1 + 3 \cdot c_2 \\\\
$$

We fill in \\(T\left(\frac{n}{4}\right)\\), where \\(T\left(\frac{n}{4}\right) = 2 \cdot T\left(\frac{n}{8}\right) + n \cdot c_1 + c_2\\)
$$
T(n) = 4 \cdot (2 \cdot T\left(\frac{n}{8}\right) + n \cdot c_1 + c_2) + 3n \cdot c_1 + 3 \cdot c_2 \\\\
T(n) = 8 \cdot T\left(\frac{n}{8}\right) + 7n \cdot c_1 + 7 \cdot c_2 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(n) = 2^{k} \cdot T\left(\frac{n}{2^{k}}\right) + (2^{k} - 1) \cdot n \cdot c_1 + (2^{k} - 1) \cdot c_2 \\\\
$$

To substitute out the recursive part, \\(T\left(\frac{n}{2^{k}}\right)\\), from the equation, we need to make \\(T\left(\frac{n}{2^{k}}\right)\\) equal to the base case, \\(T(1)\\).
Which means that we need to solve the equation \\(\frac{n}{2^{k}} = 1\\) for \\(k\\).

$$
\frac{n}{2^{k}} = 1 \\\\
2^{k} = n \\\\
k = \log(n)
$$

So we can substitute \\(k\\) with \\(\log(n)\\).

$$
T(n) = 2^{\log(n)} \cdot T\left(\frac{n}{2^{\log(n)}}\right) + (2^{\log(n)} - 1) \cdot n \cdot c_1 + (2^{\log(n)} - 1) \cdot c_2 \\\\
T(n) = n \cdot T\left(\frac{n}{n}\right) + (n - 1) \cdot n \cdot c_1 + (n - 1) \cdot c_2 \\\\
T(n) = n \cdot T(1) + (n^{2} - n) \cdot c_1 + (n - 1) \cdot c_2 \\\\
T(n) = n \cdot c_0 + (n^{2} - n) \cdot c_1 + (n - 1) \cdot c_2 \\\\
$$

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n ^{2} + n\\), which is a quadratic growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(n^{2})\\).