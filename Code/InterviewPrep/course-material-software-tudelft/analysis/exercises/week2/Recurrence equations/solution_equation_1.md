For this exercise we only need to give the closed form and the corresponding complexity in Big-Oh notation. In this case, you are allowed to get the closed form by making an educated guess. For clarity, the answer includes an explanation of how we can derive the closed form, by repeatedly unfolding. 

## Unfolding
We first derive the closed form by repeatedly unfolding.

$$
T(0) = c_0 \\\\
T(n) = T(n - 1) + n c_1 \\\\
$$

We fill in \\(T(n - 1)\\), where \\(T(n - 1) = T(n - 2) + n c_1\\)
$$
T(n) = T(n - 2) + n c_1 + n c_1 \\\\
$$

We fill in \\(T(n - 2)\\), where \\(T(n - 2) = T(n - 3) + n c_1\\)
$$
T(n) = T(n - 3) + n c_1 + n c_1 + n c_1 \\\\
T(n) = T(n - 3) + 3 n c_1 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(n) = T(n - k) + k n c_1
$$

To substitute out the recursive part, \\(T(n - k)\\), from the equation, we need to make \\(T(n - k )\\) equal to the base case, \\(T(0)\\).
Which means that we need to solve the equation \\(n - k = 0\\) for \\(k\\).

$$
n - k = 0 \\\\
k = n \\\\
$$

So we can substitute \\(k\\) with \\(n\\).

$$
T(n) = T(n - n) + n n c_1 \\\\
T(n) = T(0) + n^{2} c_1 \\\\
T(n) = c_0 + n^{2} c_1
$$

## Big-Oh

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n^{2}\\), which is a quadratic growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(n^{2})\\).